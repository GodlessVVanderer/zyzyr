use alloc::vec::Vec;
use core::alloc::Layout;
use core::mem::MaybeUninit;
use core::ptr;

use crate::cpu::Cpu;
use crate::xxh64::Xxh64;
use crate::{at, compressed, header, pow_field, put, Ctx, Dictionary, Error, Header, BLOCK_MAX, HEAD_RESERVED, MAGIC, SKIP_MAGIC, SLACK};

/// Copies `n` bytes from `src` to `dst`: up to 16 inline, as two loads and two stores of the widest width
/// that fits that may overlap in the middle (the feeds of a few bytes each called `memcpy` per piece),
/// longer through `copy_nonoverlapping`.
///
/// # Safety
/// As `ptr::copy_nonoverlapping(src, dst, n)`.
#[inline(always)]
unsafe fn copy(src: *const u8, dst: *mut u8, n: usize) {
    // SAFETY: by the caller; every access lies in the first `n` bytes of `src` or `dst`.
    unsafe {
        if n < 2 {
            if n == 1 {
                dst.write(src.read());
            }
        } else if n < 4 {
            let (a, b) = (src.cast::<u16>().read_unaligned(), src.add(n - 2).cast::<u16>().read_unaligned());
            dst.cast::<u16>().write_unaligned(a);
            dst.add(n - 2).cast::<u16>().write_unaligned(b);
        } else if n < 8 {
            let (a, b) = (src.cast::<u32>().read_unaligned(), src.add(n - 4).cast::<u32>().read_unaligned());
            dst.cast::<u32>().write_unaligned(a);
            dst.add(n - 4).cast::<u32>().write_unaligned(b);
        } else if n <= 16 {
            let (a, b) = (src.cast::<u64>().read_unaligned(), src.add(n - 8).cast::<u64>().read_unaligned());
            dst.cast::<u64>().write_unaligned(a);
            dst.add(n - 8).cast::<u64>().write_unaligned(b);
        } else {
            ptr::copy_nonoverlapping(src, dst, n);
        }
    }
}

/// The first `N` bytes of a phase's piece, little-endian, unchecked: `pump` hands each phase exactly its
/// `need` bytes, which is at least what the phase reads (4 for the magic, skip size and checksum, 3 for a block
/// header, 1 for the descriptor). Indexed, the bounds checks' panic calls kept a frame record in `process` on
/// aarch64.
///
/// # Safety
/// `e.len() >= N`, `N <= 4`.
#[inline(always)]
unsafe fn le<const N: usize>(e: &[u8]) -> u32 {
    // SAFETY: by the caller.
    unsafe { (0..N).fold(0, |v, i| v | (*e.get_unchecked(i) as u32) << (8 * i)) }
}

/// Marks the path that calls it as the less frequent one for code layout (`core::hint::cold_path` is not
/// stable): `pump` places a whole piece read in place (once per block or header) after the buffered piece (every
/// feed of a few bytes).
#[cold]
#[inline(always)]
fn rare() {}

/// A window larger than `isize::MAX` bytes (as `Vec`'s own capacity check).
#[cold]
#[inline(never)]
fn overflow() -> ! {
    panic!("capacity overflow")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Magic,
    Fhd,
    Header,
    BlockHead,
    Block { kind: u8, last: bool },
    Checksum,
    SkipSize,
    Skip,
    Failed(Error),
}

/// The frame being decoded. The hasher is always present and `check` says whether the frame has a
/// checksum: an `Option<Xxh64>` chosen by the header was built on the stack and copied in.
struct Frame {
    hasher: Xxh64,
    check: bool,
    fcs: Option<u64>,
    window: usize,
    block_max: usize,
    total: u64,
}

/// Work block of `pump`: the input of the current feed, the bytes of it used and the dictionary (0 for
/// none), so the feed loop holds the decoder alone across its calls.
#[derive(Clone, Copy, Default)]
struct Feed {
    src: usize,
    len: usize,
    used: usize,
    dict: usize,
}

/// A streaming decoder: feed it input in pieces of any size and read decoded bytes as they become ready.
///
/// Call [`feed`](Self::feed) with the next input, take output with [`pending`](Self::pending) and
/// [`consume`](Self::consume) (or [`drain`](Self::drain)), and call [`finish`](Self::finish) at the end
/// of the input to confirm the last frame was complete.
// `repr(C)` in the field order the decoder had before the header and context fields were added (those
// last), so the fields every feed touches (phase, feed block, `need`, `wp`, `lo`) stay at their offsets,
// together in 72 bytes.
#[repr(C)]
pub struct StreamDecoder {
    ext: Option<(usize, usize)>,
    frame: Option<Frame>,
    inb: Vec<u8>,
    win: Vec<u8>,
    /// The context, made by `make` once per decoder (`made`), dropped by `drop`.
    ctx: MaybeUninit<Ctx>,
    phase: Phase,
    feed: Feed,
    need: usize,
    fill: usize,
    wp: usize,
    lo: usize,
    skip: u64,
    cpu: Cpu,
    made: bool,
    /// The frame header, descriptor first (at most 14 bytes), and what `header` read from it: `begin`
    /// takes the decoder alone.
    hb: [u8; 16],
    head: MaybeUninit<Header>,
}

impl Drop for StreamDecoder {
    fn drop(&mut self) {
        if self.made {
            // SAFETY: `made`: `make` wrote the context, dropped only here.
            unsafe { self.ctx.assume_init_drop() };
        }
    }
}

impl Default for StreamDecoder {
    fn default() -> Self {
        StreamDecoder::new()
    }
}

impl StreamDecoder {
    /// A new stream decoder, using the fastest code path this CPU supports.
    pub fn new() -> Self {
        StreamDecoder::on(Cpu::detect())
    }

    pub(crate) fn on(cpu: Cpu) -> Self {
        StreamDecoder { feed: Feed::default(), hb: [0; 16], head: MaybeUninit::uninit(), ctx: MaybeUninit::uninit(), made: false, cpu, phase: Phase::Magic, need: 4, fill: 0, inb: Vec::new(), frame: None, win: Vec::new(), wp: 0, lo: 0, ext: None, skip: 0 }
    }

    /// Discards any partial frame and pending output, ready for a new stream.
    pub fn reset(&mut self) {
        self.phase = Phase::Magic;
        self.need = 4;
        self.inb.clear();
        self.frame = None;
        self.wp = 0;
        self.lo = 0;
        self.ext = None;
        self.skip = 0;
    }

    /// Feeds input to the decoder. Returns how many bytes of `input` were taken; feed the rest again
    /// after taking the pending output.
    pub fn feed(&mut self, input: &[u8]) -> Result<usize, Error> {
        self.feed_inner(input, None)
    }

    /// [`feed`](Self::feed) for frames compressed with a dictionary.
    pub fn feed_with_dict(&mut self, input: &[u8], dict: &Dictionary) -> Result<usize, Error> {
        self.feed_inner(input, Some(dict))
    }

    /// Decoded bytes ready to be read. Mark them read with [`consume`](Self::consume).
    pub fn pending(&self) -> &[u8] {
        &self.win[self.lo..self.wp]
    }

    /// Marks `n` bytes of [`pending`](Self::pending) as read.
    pub fn consume(&mut self, n: usize) {
        self.lo += n.min(self.wp - self.lo);
    }

    /// Copies pending output into `out` and marks it read. Returns the number of bytes copied.
    pub fn drain(&mut self, out: &mut [u8]) -> usize {
        let n = out.len().min(self.wp - self.lo);
        out[..n].copy_from_slice(&self.win[self.lo..self.lo + n]);
        self.lo += n;
        n
    }

    /// True when no frame is partly decoded and all output has been read.
    pub fn at_frame_boundary(&self) -> bool {
        self.phase == Phase::Magic && self.inb.is_empty() && self.lo == self.wp
    }

    /// Checks the end of the input: an error if a frame is incomplete or decoding failed.
    pub fn finish(&self) -> Result<(), Error> {
        match self.phase {
            Phase::Failed(e) => Err(e),
            _ if self.at_frame_boundary() => Ok(()),
            _ => Err(Error::Truncated),
        }
    }

    /// How many more input bytes the decoder needs to make progress (0 when it needs none).
    pub fn input_hint(&self) -> usize {
        match self.phase {
            Phase::Magic if self.inb.is_empty() => 0,
            Phase::Skip => self.skip.min(BLOCK_MAX as u64) as usize,
            Phase::Failed(_) => 0,
            _ => self.need - self.inb.len(),
        }
    }

    /// Bytes of buffer memory the decoder currently holds.
    pub fn memory(&self) -> usize {
        self.win.capacity() + self.inb.capacity()
    }

    /// Stores the dictionary in the feed block and runs `pump`: a tail call on aarch64 and wasm32; on x86_64
    /// the public entry points keep only the result's address across it (Win64 returns it in `rax`).
    #[inline(always)]
    pub(crate) fn feed_inner(&mut self, input: &[u8], dict: Option<&Dictionary>) -> Result<usize, Error> {
        put(&mut self.feed.dict, dict.map_or(0, |d| ptr::from_ref(d) as usize));
        self.pump(input)
    }

    /// Feeds `input` with the dictionary in the feed block, until the input is used up, output is pending
    /// or `process` fails (the decoder then fails on its error); the bytes used, or the error.
    ///
    /// The loop holds the decoder and the result's address alone across `process`, the allocation and the
    /// copy: the input and the dictionary are in the block, read where used, and the position and the
    /// length are registers between calls, the position stored before each call and both read back after
    /// it. (With the input, its position and the dictionary held, and the input buffer taken out of the
    /// decoder for `process`, which kept its address for an unwind path that frees it, the loop took 56
    /// stack moves on x86_64. With the position kept only in the block, each feed's result waited on four
    /// store-to-load forwards in a row: 1-byte feeds 16% slower.)
    #[inline(never)]
    fn pump(&mut self, input: &[u8]) -> Result<usize, Error> {
        if let Phase::Failed(_) = self.phase {
            return self.fail();
        }
        put(&mut self.feed.src, input.as_ptr() as usize);
        put(&mut self.feed.len, input.len());
        let (mut used, mut len) = (0, input.len());
        while self.lo == self.wp && used < len {
            if self.phase == Phase::Skip {
                put(&mut self.feed.used, used);
                self.skipping();
                (used, len) = (at(&self.feed.used), at(&self.feed.len));
                continue;
            }
            // The buffered piece first: feeds of a few bytes take it every time, and as the `else` arm it was
            // reached by a taken branch (as was the rest past the skip test, now behind a cold call).
            let ok = if !self.inb.is_empty() || len - used < self.need {
                if self.inb.capacity() < self.need {
                    put(&mut self.feed.used, used);
                    self.grow();
                    (used, len) = (at(&self.feed.used), at(&self.feed.len));
                }
                // The length is set before the copy and the copy's arguments are its last values, so
                // nothing waits through it (`extend_from_slice` held the length and the count). `need` is
                // read here again: read once above, it waited through the allocation on aarch64.
                let have = self.inb.len();
                let take = (at(&self.need) - have).min(len - used);
                put(&mut self.feed.used, used + take);
                // SAFETY: `have + take <= need <= capacity`; the bytes are written by the copy below before
                // anything reads them; `used + take <= len` of the live input bound by `feed_inner`.
                unsafe {
                    self.inb.set_len(have + take);
                    copy((at(&self.feed.src) + used) as *const u8, self.inb.as_mut_ptr().add(have), take);
                }
                if self.inb.len() != self.need {
                    (used, len) = (at(&self.feed.used), at(&self.feed.len));
                    continue;
                }
                // SAFETY: the buffered bytes in their own allocation, which `process` neither frees nor
                // writes (it touches no `inb`); cleared after.
                let e = unsafe { core::slice::from_raw_parts(self.inb.as_ptr(), self.inb.len()) };
                let ok = self.process(e);
                self.inb.clear();
                ok
            } else {
                rare();
                let n = self.need;
                put(&mut self.feed.used, used + n);
                // SAFETY: `used + n <= len`; `feed_inner` bound the live input.
                let e = unsafe { core::slice::from_raw_parts((at(&self.feed.src) + used) as *const u8, n) };
                self.process(e)
            };
            if !ok {
                return self.fail();
            }
            (used, len) = (at(&self.feed.used), at(&self.feed.len));
        }
        Ok(used)
    }

    /// The error the decoder failed on (in its phase), as the result. Out of line, so `pump` writes no error
    /// result itself: its error and success exits stored their result tags in one merged block, the success
    /// tag held in a callee-saved register through the loop.
    #[inline(never)]
    #[cold]
    fn fail(&mut self) -> Result<usize, Error> {
        match self.phase {
            Phase::Failed(e) => Err(e),
            _ => Err(Error::Truncated),
        }
    }

    /// The decoder fails on `e`; false, the result of the step that found it. Each step returns a flag and
    /// writes its error here as an immediate: a `Result` from a step with several calls was merged through
    /// one return block, so no call was a tail call.
    #[inline(always)]
    fn bad(&mut self, e: Error) -> bool {
        self.phase = Phase::Failed(e);
        false
    }

    /// The skipped bytes of a skippable frame the feed holds. A leaf: its stores of the next phase's
    /// constants, in `pump`, held them in two callee-saved registers through the loop on aarch64. Cold: skippable
    /// frames are rare, and the feed loop's own path follows the skip test without a taken branch.
    #[inline(never)]
    #[cold]
    fn skipping(&mut self) {
        let used = self.feed.used;
        let n = ((self.feed.len - used) as u64).min(self.skip) as usize;
        self.feed.used = used + n;
        self.skip -= n as u64;
        if self.skip == 0 {
            self.phase = Phase::Magic;
            self.need = 4;
        }
    }

    /// The input buffer grown to exactly `need` bytes, its bytes kept, by the allocator directly: through
    /// `reserve_exact` the length and `need` waited across the allocation and its result came back through
    /// the stack. `need` is read again after the call. Out of line: inlined into `pump`, a program's own
    /// allocator came with it, and its function addresses were hoisted into callee-saved registers that
    /// `pump` saved on every feed; the first allocation's `need` also waited through the allocator shim
    /// that `alloc` calls first.
    #[inline(never)]
    #[cold]
    fn grow(&mut self) {
        let cap = self.inb.capacity();
        // SAFETY: `cap < need`, so `0 < need`; a `Vec<u8>` with a capacity holds a global allocation of `cap`
        // bytes, which `realloc` moves; the vector is rebuilt over the new allocation with its length (written
        // without dropping the old one, which no longer exists or, with no capacity, owns nothing).
        unsafe {
            let new = if cap == 0 {
                alloc::alloc::alloc(Layout::from_size_align_unchecked(self.need, 1))
            } else {
                alloc::alloc::realloc(self.inb.as_mut_ptr(), Layout::from_size_align_unchecked(cap, 1), self.need)
            };
            let need = at(&self.need);
            if new.is_null() {
                alloc::alloc::handle_alloc_error(Layout::from_size_align_unchecked(need, 1));
            }
            ptr::write(&mut self.inb, Vec::from_raw_parts(new, self.inb.len(), need));
        }
    }

    /// One piece of `need` bytes for the current phase; false when the decoder failed (the error in its
    /// phase). The header phases are short and inline; the frame start, the block and the checksum are tail
    /// calls, so nothing waits across a call here. (With the block inlined, its window growth, copies, block
    /// decode and hash, and the checksum's 5-argument call, it kept 8 callee-saved pairs and 3 stack slots: 27
    /// stack moves on x86_64.)
    fn process(&mut self, e: &[u8]) -> bool {
        match self.phase {
            Phase::Magic => {
                // SAFETY (each `le` here): `e` is the phase's `need` bytes (see `le`).
                let m = unsafe { le::<4>(e) };
                if m == MAGIC {
                    self.phase = Phase::Fhd;
                    self.need = 1;
                } else if m & 0xFFFF_FFF0 == SKIP_MAGIC {
                    self.phase = Phase::SkipSize;
                    self.need = 4;
                } else {
                    return self.bad(Error::BadMagic);
                }
            }
            Phase::SkipSize => {
                self.skip = unsafe { le::<4>(e) } as u64;
                self.phase = if self.skip == 0 { Phase::Magic } else { Phase::Skip };
                self.need = 4;
            }
            Phase::Fhd => {
                let fhd = unsafe { le::<1>(e) } as u8;
                let single = fhd & 0x20 != 0;
                // 0, 1, 2, 4 and 2, 4, 8 by arithmetic: the table was built on the stack, the `match` a jump.
                let did = pow_field((fhd & 3) as usize);
                let code = (fhd >> 6) as usize;
                let fcs = if code == 0 { single as usize } else { 2 * pow_field(code) };
                let rest = (!single) as usize + did + fcs;
                self.hb[0] = fhd;
                if rest == 0 {
                    return self.begin();
                }
                self.phase = Phase::Header;
                self.need = rest;
            }
            Phase::Header => {
                // The header after its descriptor, at most 13 bytes (`need` from the descriptor), inline.
                let n = e.len().min(15);
                // SAFETY: `n <= e.len()` and `1 + n <= 16`.
                unsafe { copy(e.as_ptr(), self.hb.as_mut_ptr().add(1), n) };
                return self.begin();
            }
            Phase::BlockHead => {
                let bh = unsafe { le::<3>(e) } as usize;
                let kind = ((bh >> 1) & 3) as u8;
                let size = bh >> 3;
                let Some(f) = self.frame.as_ref() else { return self.bad(Error::Truncated) };
                if size > f.block_max {
                    return self.bad(Error::BlockSize);
                }
                if kind == 3 {
                    return self.bad(Error::BlockType);
                }
                self.phase = Phase::Block { kind, last: bh & 1 != 0 };
                self.need = if kind == 1 { 1 } else { size };
                self.fill = size;
                if self.need == 0 {
                    return self.block(&[]);
                }
            }
            Phase::Block { .. } => return self.block(e),
            Phase::Checksum => return self.verify(e),
            Phase::Skip | Phase::Failed(_) => {}
        }
        true
    }

    /// The frame's checksum `e` (4 bytes) against the hash of its content; the frame ends.
    #[inline(never)]
    fn verify(&mut self, e: &[u8]) -> bool {
        // SAFETY: `e` is the phase's 4 bytes (see `le`).
        let want = unsafe { le::<4>(e) };
        let Some(f) = self.frame.as_ref() else { return self.bad(Error::Truncated) };
        if f.hasher.finish() as u32 != want {
            return self.bad(Error::Checksum);
        }
        self.end();
        true
    }

    fn end(&mut self) {
        self.frame = None;
        self.phase = Phase::Magic;
        self.need = 4;
    }

    /// Starts the frame whose header is in `hb`, with the dictionary in the feed block. The header is read
    /// into `head` and its fields are read from there after each call, so the decoder is the only value
    /// held across `header`, the context's construction and its reset. (With the header bytes and the
    /// dictionary as arguments, the header in a stack local, the hasher an `Option` built on the stack and
    /// the context made inline, it kept 8 callee-saved pairs and 4 vector and 5 general stack moves.)
    fn begin(&mut self) -> bool {
        if header(&self.hb, &mut self.head) == HEAD_RESERVED {
            return self.bad(Error::ReservedBit);
        }
        // SAFETY: `header` wrote every field: `hb` holds the whole header (16 bytes, the longest is 14), so it
        // returned a length or `HEAD_RESERVED`.
        let did = unsafe { self.head.assume_init_ref() }.did;
        if did != 0 && self.dict().is_none_or(|d| d.id != did) {
            return self.bad(Error::Dictionary(did));
        }
        if !self.made {
            self.make();
        }
        let dict = self.dict();
        // SAFETY: `make` wrote the context.
        unsafe { self.ctx.assume_init_mut() }.reset(dict);
        // SAFETY: as above.
        let h = unsafe { self.head.assume_init_ref() };
        let block_max = h.window.min(BLOCK_MAX as u64) as usize;
        let window = usize::try_from(h.window).unwrap_or(usize::MAX);
        self.frame = Some(Frame { hasher: Xxh64::new(0), check: h.checksum, fcs: h.fcs(), window, block_max, total: 0 });
        self.wp = 0;
        self.lo = 0;
        self.ext = None;
        self.phase = Phase::BlockHead;
        self.need = 3;
        true
    }

    /// The dictionary bound in the feed block, live for the feed (not borrowed from the decoder).
    #[inline(always)]
    fn dict<'a>(&self) -> Option<&'a Dictionary> {
        // SAFETY: `feed_inner` bound a live dictionary or 0 for the whole feed.
        unsafe { (at(&self.feed.dict) as *const Dictionary).as_ref() }
    }

    /// The context, made once per decoder. Cold and out of line, so `begin` holds nothing for it. It is
    /// built by `Ctx::new` and copied into its slot (352 bytes, once): written in place by a second
    /// constructor, which also referred to the sequence state's constructor, a program placed that
    /// function between the frame loop and the block decoder, moving every hot function after it (the
    /// 1 KB-block file 0.8% slower with every function on a 4 KB boundary, the decoder code identical).
    #[inline(never)]
    #[cold]
    fn make(&mut self) {
        // SAFETY: not `made`, so the slot holds nothing to drop; `made` is set once it is whole.
        unsafe { self.ctx.as_mut_ptr().write(Ctx::new(self.cpu)) };
        self.made = true;
    }

    /// Room for a block at `wp` (moved to the window's start, the history kept above it, when the window would
    /// pass its limit): the end of the room. The growth is in `widen`, so only the window, the block size and
    /// the limit are live here.
    #[inline(always)]
    fn room(&mut self, window: usize, need: usize) -> usize {
        let limit = window.saturating_add(2 * need);
        if self.wp + need > limit {
            self.ext = Some((self.wp - window, self.wp));
            self.wp = 0;
            self.lo = 0;
        }
        if let Some((_, hi)) = self.ext {
            if self.wp >= window {
                self.ext = None;
            } else {
                return hi - (window - self.wp);
            }
        }
        if self.win.capacity() < self.wp + need {
            self.widen();
        }
        self.win.capacity()
    }

    /// The window grown for a block at `wp`: doubled, at least to the block's end, at most to the window's
    /// limit or the frame's content size; its bytes kept. Cold and out of line, reading the frame itself:
    /// inlined, the content size and the target's terms waited in callee-saved registers of the block. By the
    /// allocator directly, as `grow`: through `reserve_exact` the allocation's result came back through the
    /// stack.
    #[inline(never)]
    #[cold]
    fn widen(&mut self) {
        let Some(f) = self.frame.as_ref() else { return };
        let need = f.block_max + SLACK;
        let limit = f.window.saturating_add(2 * need);
        let want = self.wp + need;
        let cap = self.win.capacity();
        let mut target = cap.saturating_mul(2).max(want).min(limit.max(want));
        if let Some(n) = f.fcs {
            let whole = usize::try_from(n).unwrap_or(usize::MAX).saturating_add(need);
            if whole >= want {
                target = target.min(whole);
            }
        }
        let Ok(layout) = Layout::array::<u8>(target) else { overflow() };
        // SAFETY: `cap < want <= target`, so `0 < target`, within `isize::MAX` (`layout`); a `Vec<u8>` with a
        // capacity holds a global allocation of `cap` bytes, which `realloc` moves; the vector is rebuilt over
        // the new allocation with its length (written without dropping the old one, which no longer exists or,
        // with no capacity, owns nothing).
        unsafe {
            let new = if cap == 0 { alloc::alloc::alloc(layout) } else { alloc::alloc::realloc(self.win.as_mut_ptr(), Layout::from_size_align_unchecked(cap, 1), target) };
            if new.is_null() {
                alloc::alloc::handle_alloc_error(layout);
            }
            ptr::write(&mut self.win, Vec::from_raw_parts(new, self.win.len(), target));
        }
    }

    /// The block `body` of the current `Block` phase decoded into the window, then the next phase. Out of
    /// line, a tail call of `process`. The body waits in the context's job, where `compressed` reads it, and
    /// every value is read from the decoder after each call, so the decoder alone is held: the window's growth,
    /// the copy, the fill or the block decode, then the phase change, and the hash last.
    #[inline(never)]
    fn block(&mut self, body: &[u8]) -> bool {
        // `begin` made the context before the first block header; checked so the job may be written.
        if !self.made {
            return self.bad(Error::Truncated);
        }
        {
            // SAFETY: `made`.
            let j = &mut unsafe { self.ctx.assume_init_mut() }.job;
            put(&mut j.blk, body.as_ptr() as usize);
            put(&mut j.len, body.len());
        }
        let Some(f) = self.frame.as_ref() else { return self.bad(Error::Truncated) };
        let cap = self.room(f.window, f.block_max + SLACK);
        // SAFETY (both reads of the phase): `process` runs a block only in the `Block` phase, and nothing here
        // changes the phase before the reads. Read where used, not held across the calls; tested again after
        // them, the variant test took one more callee-saved register.
        let Phase::Block { kind, .. } = at(&self.phase) else { unsafe { core::hint::unreachable_unchecked() } };
        let base = self.win.as_mut_ptr();
        // SAFETY: `made`; the job's block is `body`, live for the call. `room` left at least `block_max + SLACK`
        // bytes of capacity from `wp` up to `cap`, so a raw block (`len <= block_max`) or a fill (`fill <=
        // block_max`) lies inside the window.
        unsafe {
            if kind == 0 {
                let j = &self.ctx.assume_init_ref().job;
                ptr::copy_nonoverlapping(at(&j.blk) as *const u8, base.add(at(&self.wp)), at(&j.len));
                let end = at(&self.wp) + at(&self.ctx.assume_init_ref().job.len);
                put(&mut self.wp, end);
            } else if kind == 1 {
                let j = &self.ctx.assume_init_ref().job;
                ptr::write_bytes(base.add(at(&self.wp)), (at(&j.blk) as *const u8).read(), at(&self.fill));
                let end = at(&self.wp) + at(&self.fill);
                put(&mut self.wp, end);
            } else {
                let ring = self.ext.is_some();
                let ext: &[u8] = match self.ext {
                    Some((_, hi)) => core::slice::from_raw_parts(base.add(cap), hi - cap),
                    None => self.dict().map_or(&[], |d| &d.content),
                };
                let Some(f) = self.frame.as_ref() else { return self.bad(Error::Truncated) };
                let (window, block_max, wp) = (f.window, f.block_max, self.wp);
                // The rest of the job's inputs (every other field is written by `compressed` before it is read);
                // binding the block again from volatile reads of the job held them in a callee-saved register.
                let ctx = self.ctx.assume_init_mut();
                let j = &mut ctx.job;
                (j.base, j.pos, j.cap) = (base as usize, wp, cap);
                (j.ext, j.ext_len, j.fs, j.window, j.block_max, j.ring) = (ext.as_ptr() as usize, ext.len(), 0, window, block_max, ring);
                if !compressed(ctx) {
                    let e = ctx.job.err.0;
                    return self.bad(e);
                }
                put(&mut self.wp, at(&self.ctx.assume_init_ref().job.pos));
            }
        }
        // The block's bytes are `lo .. wp`: `pump` runs a block only when `lo == wp`, and `room` moves both.
        let (start, end) = (at(&self.lo), at(&self.wp));
        if end > self.win.len() {
            // SAFETY: the block wrote `start .. end` inside the capacity.
            unsafe { self.win.set_len(end) };
        }
        let Some(f) = self.frame.as_mut() else { return self.bad(Error::Truncated) };
        f.total += (end - start) as u64;
        if let Some(n) = f.fcs {
            if f.total > n {
                return self.bad(Error::ContentSize);
            }
        }
        let check = f.check;
        let Phase::Block { last, .. } = at(&self.phase) else { unsafe { core::hint::unreachable_unchecked() } };
        if last {
            let Some(f) = self.frame.as_ref() else { return self.bad(Error::Truncated) };
            if let Some(n) = f.fcs {
                if f.total != n {
                    return self.bad(Error::ContentSize);
                }
            }
            if check {
                self.phase = Phase::Checksum;
                self.need = 4;
            } else {
                self.end();
            }
        } else {
            self.phase = Phase::BlockHead;
            self.need = 3;
        }
        if check {
            self.hash();
        }
        true
    }

    /// The block `lo .. wp` hashed into the frame's checksum. A leaf (the hasher's buffer copies are inline);
    /// the window slice unchecked, as its bounds check's panic call kept a frame record on aarch64.
    #[inline(never)]
    fn hash(&mut self) {
        let (lo, wp) = (self.lo, self.wp);
        if let Some(f) = self.frame.as_mut() {
            // SAFETY: `block` leaves `lo <= wp <= win.len()` (it sets the length to `wp` when the block passes it).
            f.hasher.update(unsafe { self.win.get_unchecked(lo..wp) });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_every_length() {
        let src: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37) ^ 0x5A).collect();
        for n in 0..=40 {
            for (s, d) in [(0, 0), (1, 3), (7, 5)] {
                let mut dst = [0xEEu8; 64];
                // SAFETY: `s + n` and `d + n` are within both 64-byte buffers.
                unsafe { copy(src.as_ptr().add(s), dst.as_mut_ptr().add(d), n) };
                assert_eq!(&dst[d..d + n], &src[s..s + n], "n {n}");
                assert!(dst[..d].iter().chain(&dst[d + n..]).all(|&b| b == 0xEE), "n {n} wrote outside");
            }
        }
    }
}
