//! A Zstandard (RFC 8878) decompressor in pure Rust, with no dependencies.
//!
//! The core is `no_std` and needs only `alloc`. It picks the fastest code path for the CPU at runtime,
//! and an optional C API (feature `capi`) and libzstd-compatible API (feature `compat`) are available.
//!
//! # One-shot decoding
//!
//! ```
//! # let compressed = zstd::encode_all(&b"hello, zstd"[..], 3).unwrap();
//! let data = zyzyr::decompress(&compressed)?;
//! assert_eq!(data, b"hello, zstd");
//! # Ok::<(), zyzyr::Error>(())
//! ```
//!
//! # Reusing a decoder
//!
//! A [`Decoder`] keeps its working memory between calls, which is faster when decoding many frames.
//! [`Decoder::decompress_into`] appends to an existing vector.
//!
//! ```
//! # let compressed = zstd::encode_all(&b"hello, zstd"[..], 3).unwrap();
//! let mut decoder = zyzyr::Decoder::new();
//! let mut out = Vec::new();
//! let n = decoder.decompress_into(&compressed, &mut out)?;
//! assert_eq!(&out[..n], b"hello, zstd");
//! # Ok::<(), zyzyr::Error>(())
//! ```
//!
//! # Streaming
//!
//! A [`StreamDecoder`] takes input in pieces of any size and gives back decoded bytes as they are ready,
//! with memory bounded by the frame's window.
//!
//! # Dictionaries
//!
//! Frames compressed with a dictionary are decoded with the `*_with_dict` functions and a [`Dictionary`].
//!
//! # Features
//!
//! - `std` (default): links the standard library. Disable it for `no_std` targets.
//! - `capi`: a C API, with the header in `include/zyzyr.h`.
//! - `compat`: a drop-in libzstd-compatible C API (`ZSTD_decompress`, `ZSTD_decompressStream`, ...).

#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod bits;
#[cfg(feature = "capi")]
pub mod capi;
#[cfg(feature = "compat")]
pub mod compat;
mod cpu;
mod fse;
mod huf;
mod seq;
mod stream;
mod xxh64;

pub use stream::StreamDecoder;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::alloc::Layout;
use core::ffi::CStr;
use core::fmt;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ptr;

use cpu::Cpu;
use huf::HufTable;
use seq::SeqState;
use xxh64::Xxh64;

const MAGIC: u32 = 0xFD2F_B528;
const DICT_MAGIC: u32 = 0xEC30_A437;
const BLOCK_MAX: usize = 128 * 1024;
pub(crate) const SLACK: usize = 32;

/// Why decoding failed. [`Error::text`] and `Display` give a short description.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Truncated,
    BadMagic,
    ReservedBit,
    Dictionary(u32),
    DictionaryFormat,
    BlockType,
    BlockSize,
    Literals,
    Huffman,
    Fse,
    Bitstream,
    Sequences,
    Offset,
    ContentSize,
    Checksum,
}

impl Error {
    /// A short description of the error, as a C string so the C API can return it directly.
    pub const fn text(self) -> &'static CStr {
        match self {
            Error::Truncated => c"input ends before the data it describes",
            Error::BadMagic => c"not a Zstandard or skippable frame",
            Error::ReservedBit => c"reserved frame header bit is set",
            Error::Dictionary(_) => c"frame needs a dictionary that was not supplied",
            Error::DictionaryFormat => c"malformed dictionary",
            Error::BlockType => c"reserved block type",
            Error::BlockSize => c"block larger than the frame's block maximum",
            Error::Literals => c"malformed literals section",
            Error::Huffman => c"malformed Huffman table or stream",
            Error::Fse => c"malformed FSE table description",
            Error::Bitstream => c"bitstream without its end marker",
            Error::Sequences => c"malformed sequences section",
            Error::Offset => c"match offset outside the decoded data or window",
            Error::ContentSize => c"decoded size differs from the frame content size",
            Error::Checksum => c"content checksum mismatch",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Dictionary(id) => write!(f, "frame needs dictionary {id}, which was not supplied"),
            e => f.write_str(e.text().to_str().unwrap_or("")),
        }
    }
}

impl core::error::Error for Error {}

/// The job first, so the context and its work block are one pointer.
#[repr(C)]
struct Ctx {
    job: Job,
    lits: Vec<u8>,
    huf: Box<HufTable>,
    huf_valid: bool,
    seq: Box<SeqState>,
    cpu: Cpu,
    /// The checksum of the frame `blocks` decodes, hashed over the output where it lies.
    sum: Xxh64,
    /// The header of the frame `frame` decodes, written there by `header`; last, so no field before it
    /// moves.
    head: MaybeUninit<Header>,
}

/// Work block of `compressed`: the block, the output and its history, and where the literals section
/// left the literals, so the block decoder holds only the context across its calls and every phase
/// reads its inputs from here.
#[derive(Clone, Copy, Default)]
pub(crate) struct Job {
    pub(crate) blk: usize,
    pub(crate) len: usize,
    pub(crate) base: usize,
    pub(crate) pos: usize,
    pub(crate) cap: usize,
    pub(crate) ext: usize,
    pub(crate) ext_len: usize,
    pub(crate) fs: usize,
    pub(crate) window: usize,
    pub(crate) block_max: usize,
    /// The literals: inside the block when at least `SLACK` bytes of it follow them (the copies read up
    /// to `SLACK` bytes past the last literal), otherwise in `Ctx::lits`.
    pub(crate) lit: usize,
    pub(crate) lit_len: usize,
    /// Bytes of the block the literals section takes.
    pub(crate) used: usize,
    /// The source of the step after the literals header: the raw literals to copy, the fill byte, or
    /// the Huffman section and its length; and whether the Huffman section holds four streams.
    pub(crate) src: usize,
    pub(crate) src_len: usize,
    pub(crate) four: bool,
    pub(crate) ring: bool,
    /// The error of a `compressed`, `frame` or `run` that failed. Every failure writes it, so it is read
    /// without a fallback (an `Option` read with one held the fallback's tag in a callee-saved register
    /// across the call).
    pub(crate) err: Fault,
    /// The frame `blocks` reads: the input, the position of the next block header, the output vector,
    /// the frame content size (when `sized`), whether the frame has a checksum, the output index hashed
    /// up to, and whether the block just read is the last.
    pub(crate) inp: usize,
    pub(crate) inp_len: usize,
    pub(crate) ip: usize,
    pub(crate) out: usize,
    pub(crate) fcs: u64,
    pub(crate) hashed: usize,
    pub(crate) sized: bool,
    pub(crate) check: bool,
    pub(crate) last: bool,
}

/// `Job::err`: an error with a default, so the job keeps its derived default.
#[derive(Clone, Copy)]
pub(crate) struct Fault(pub(crate) Error);

impl Default for Fault {
    fn default() -> Self {
        Fault(Error::Truncated)
    }
}

impl Ctx {
    /// The buffers are held without drop glue until the context is whole: a box or vector alive across
    /// a later allocation kept its address in a stack slot for an unwind path that frees it. Allocated
    /// in the order the context always had (literals, Huffman table, sequence state), so the decode
    /// runs on the same heap layout.
    fn new(cpu: Cpu) -> Self {
        let lits = ManuallyDrop::new(vec![0; BLOCK_MAX + 2 * SLACK]);
        let huf = Box::into_raw(HufTable::boxed());
        let seq = Box::into_raw(SeqState::boxed());
        // SAFETY: `huf` and `seq` came from `Box::into_raw` above and are rebuilt once each.
        let (huf, seq) = unsafe { (Box::from_raw(huf), Box::from_raw(seq)) };
        Ctx { job: Job::default(), sum: Xxh64::new(0), lits: ManuallyDrop::into_inner(lits), huf, huf_valid: false, seq, cpu, head: MaybeUninit::uninit() }
    }

    /// A one-shot call's context, written into its slot by an out-of-line call that also detects the CPU and
    /// binds the input, so the caller does not hold it across.
    #[inline(never)]
    fn make(slot: &mut MaybeUninit<Ctx>, input: &[u8]) {
        bind(slot.write(Ctx::new(Cpu::detect())), input);
    }

    /// Drops a one-shot call's context out of line: inlined, its buffers' addresses were read from the
    /// caller's frame.
    ///
    /// # Safety
    /// `slot` holds a context made by `make`, not used again.
    #[inline(never)]
    unsafe fn free(slot: &mut MaybeUninit<Ctx>) {
        // SAFETY: by the caller.
        unsafe { slot.assume_init_drop() };
    }

    #[inline(always)]
    fn fail(&mut self, e: Error) -> bool {
        self.job.err = Fault(e);
        false
    }

    /// `fail` out of line, for a loop that calls: inline, its exits' stores merged into one store of a
    /// tag chosen on the way, and the tag's first value waited in a callee-saved register through the loop.
    /// The same happened to calls of one leaf with different errors, so the commonest error has its own.
    #[inline(never)]
    #[cold]
    fn fault(&mut self, e: Error) -> bool {
        self.fail(e)
    }

    #[inline(never)]
    #[cold]
    fn truncated(&mut self) -> bool {
        self.fail(Error::Truncated)
    }

    /// The tables, their addresses read from the context where used: the compiler knows the calls in
    /// between do not change the boxes and would hold each address in a callee-saved register.
    #[inline(always)]
    fn huf(&mut self) -> &mut HufTable {
        // SAFETY: a `Box<T>` of a sized `T` is one non-null pointer to its value, owned by `self`.
        unsafe { &mut *ptr::read((&raw const self.huf).cast::<*mut HufTable>()) }
    }

    #[inline(always)]
    fn seq(&mut self) -> &mut SeqState {
        // SAFETY: as in `huf`.
        unsafe { &mut *ptr::read((&raw const self.seq).cast::<*mut SeqState>()) }
    }

    #[inline(always)]
    fn reset(&mut self, dict: Option<&Dictionary>) {
        self.huf_valid = false;
        self.seq.reset();
        if let Some(d) = dict {
            if let Some(h) = &d.huf {
                self.huf.dt.copy_from_slice(&h.dt);
                self.huf.pair = false;
                self.huf_valid = true;
            }
            if let Some(s) = &d.seq {
                self.seq.load(s);
            }
        }
    }
}

/// A Zstandard dictionary, parsed once and shared by any number of decodes.
///
/// Accepts both formatted dictionaries (from `zstd --train`) and raw content dictionaries.
pub struct Dictionary {
    id: u32,
    content: Vec<u8>,
    huf: Option<Box<HufTable>>,
    seq: Option<Box<SeqState>>,
}

impl Dictionary {
    /// Parses a dictionary from its bytes.
    pub fn new(bytes: &[u8]) -> Result<Dictionary, Error> {
        let mut r = MaybeUninit::uninit();
        Dictionary::make(&mut r, bytes);
        // SAFETY: `make` writes the result on every path.
        unsafe { r.assume_init() }
    }

    /// Builds the dictionary in the caller's result slot, holding only the slot across the calls (the
    /// input, the id and the tables held across them took 8 callee-saved registers and 2 stack slots).
    /// Until a dictionary is written into it, the slot holds the input and the content buffer as raw
    /// words; once it is, the input lies in the sequence state's section reader and every later step reads
    /// it from there. The Huffman tree's length comes from its header byte, so the sequence tables are
    /// read before the Huffman table is allocated. Out of line, so `new` passes its own result slot
    /// (inlined, the result was built in a stack slot and copied).
    #[inline(never)]
    fn make(r: &mut MaybeUninit<Result<Dictionary, Error>>, bytes: &[u8]) {
        let Some(m) = bytes.first_chunk::<8>() else {
            r.write(Err(Error::DictionaryFormat));
            return;
        };
        let word = Dictionary::words(r);
        if u32::from_le_bytes([m[0], m[1], m[2], m[3]]) != DICT_MAGIC {
            // SAFETY: the words lie in the slot; the buffer holds `len` bytes (or is dangling for none),
            // filled from the live input before the vector is made.
            unsafe {
                put(&mut *word(0), bytes.as_ptr() as usize);
                put(&mut *word(1), bytes.len());
                put(&mut *word(2), buffer(bytes.len(), || at(&*word(1))) as usize);
                let (src, len, p) = (at(&*word(0)), at(&*word(1)), at(&*word(2)) as *mut u8);
                ptr::copy_nonoverlapping(src as *const u8, p, len);
                let (len, p) = (at(&*word(1)), at(&*word(2)) as *mut u8);
                r.write(Ok(Dictionary { id: 0, content: Vec::from_raw_parts(p, len, len), huf: None, seq: None }));
            }
            return;
        }
        if bytes.len() == 8 || u32::from_le_bytes([m[4], m[5], m[6], m[7]]) == 0 {
            r.write(Err(Error::DictionaryFormat));
            return;
        }
        // SAFETY: the words lie in the slot.
        unsafe {
            put(&mut *word(0), bytes.as_ptr() as usize);
            put(&mut *word(1), bytes.len());
        }
        let seq = SeqState::for_dict();
        // SAFETY: the words written above, read back after the allocation; the input is live for the call.
        let bytes = unsafe { core::slice::from_raw_parts(at(&*word(0)) as *const u8, at(&*word(1))) };
        let m = bytes.first_chunk::<9>().unwrap_or(&[0; 9]);
        let d = r.write(Ok(Dictionary { id: u32::from_le_bytes([m[4], m[5], m[6], m[7]]), content: Vec::new(), huf: None, seq: Some(seq) }));
        let Ok(d) = d else { unreachable!() };
        // The tree takes 1 + h bytes for an FSE-coded header byte h < 128, else one per two of h - 127
        // weights (as `huf::tree` counts it).
        let h = m[8] as usize;
        let tree = if h >= 128 { 1 + (h - 127).div_ceil(2) } else { 1 + h };
        // SAFETY: `bytes` holds at least 9 bytes (checked above).
        Dictionary::seq_of(d).bind_dict(unsafe { bytes.get_unchecked(8..) }, tree);
        if !seq::dict(|| Dictionary::seq_of(d)) {
            return Dictionary::reject(r);
        }
        // SAFETY: `huf` is `None`, written above; nothing to drop.
        unsafe { ptr::write(&raw mut d.huf, Some(HufTable::boxed())) };
        if !huf::tree(Dictionary::seq_of(d).dict_input(), Dictionary::huf_of(d)) {
            return Dictionary::reject(r);
        }
        let n = Dictionary::seq_of(d).dict_rest().len();
        // SAFETY: the buffer holds `n` bytes (or is dangling for none), the content's length read again
        // from the state where used; `content` is the empty vector written above, nothing to drop; the
        // copy fills the vector's capacity, then its length is set.
        unsafe {
            let p = buffer(n, || Dictionary::seq_of(d).dict_rest().len());
            ptr::write(&raw mut d.content, Vec::from_raw_parts(p, 0, Dictionary::seq_of(d).dict_rest().len()));
            let src = Dictionary::seq_of(d).dict_rest();
            ptr::copy_nonoverlapping(src.as_ptr(), d.content.as_mut_ptr(), src.len());
            let n = Dictionary::seq_of(d).dict_rest().len();
            d.content.set_len(n);
        }
    }

    /// The result slot as three raw words, for the values a dictionary build holds across a call before
    /// a dictionary is written over them.
    #[inline(always)]
    fn words(r: &mut MaybeUninit<Result<Dictionary, Error>>) -> impl Fn(usize) -> *mut usize {
        const _: () = assert!(size_of::<Result<Dictionary, Error>>() >= size_of::<[usize; 3]>() && align_of::<Result<Dictionary, Error>>() >= align_of::<usize>());
        let w = r.as_mut_ptr().cast::<usize>();
        // SAFETY: the slot is large and aligned enough for three words (asserted above); callers pass
        // `i < 3`.
        move |i| unsafe { w.add(i) }
    }

    /// Frees what `make` built and writes the error, out of line.
    #[inline(never)]
    #[cold]
    fn reject(r: &mut MaybeUninit<Result<Dictionary, Error>>) {
        // SAFETY: called by `make` after it wrote the dictionary into `r`.
        unsafe { r.assume_init_drop() };
        r.write(Err(Error::DictionaryFormat));
    }

    /// The tables of the dictionary `make` builds, their addresses read from it where used (as
    /// `Ctx::huf`): the compiler knows the calls in between do not change them and would hold each in a
    /// callee-saved register.
    #[inline(always)]
    fn seq_of<'a>(d: &Dictionary) -> &'a mut SeqState {
        // SAFETY: an `Option<Box<T>>` of a sized `T` is one pointer, non-null when `Some`; `make` reads it
        // after writing `Some`, and owns the state for the call.
        unsafe { &mut *ptr::read((&raw const d.seq).cast::<*mut SeqState>()) }
    }

    #[inline(always)]
    fn huf_of<'a>(d: &Dictionary) -> &'a mut HufTable {
        // SAFETY: as in `seq_of`.
        unsafe { &mut *ptr::read((&raw const d.huf).cast::<*mut HufTable>()) }
    }

    /// The dictionary id, matched against the id in a frame header (see [`dictionary_id`]).
    pub fn id(&self) -> u32 {
        self.id
    }
}

/// `n` bytes from the global allocator for a vector's buffer (dangling for none). On failure the size is
/// taken again from `again`: kept for the failure path, it waited through the allocation in a
/// callee-saved register.
///
/// # Safety
/// `n` and `again()` are the same size, at most `isize::MAX`.
#[inline(always)]
unsafe fn buffer(n: usize, again: impl FnOnce() -> usize) -> *mut u8 {
    if n == 0 {
        return ptr::NonNull::dangling().as_ptr();
    }
    // SAFETY: by the caller, a valid layout of nonzero size.
    unsafe {
        let p = alloc::alloc::alloc(Layout::from_size_align_unchecked(n, 1));
        if p.is_null() {
            alloc::alloc::handle_alloc_error(Layout::from_size_align_unchecked(again(), 1));
        }
        p
    }
}

/// Decompresses every frame in `input` into a new vector.
///
/// Concatenated frames decode as one sequence, and skippable frames are skipped.
pub fn decompress(input: &[u8]) -> Result<Vec<u8>, Error> {
    owned(input, None)
}

/// Decompresses every frame in `input`, appending to `out`. Returns the number of bytes appended.
pub fn decompress_into(input: &[u8], out: &mut Vec<u8>) -> Result<usize, Error> {
    once(input, None, out)
}

/// [`decompress`] for frames compressed with a dictionary.
pub fn decompress_with_dict(input: &[u8], dict: &Dictionary) -> Result<Vec<u8>, Error> {
    owned(input, Some(dict))
}

/// [`decompress_into`] for frames compressed with a dictionary.
pub fn decompress_into_with_dict(input: &[u8], dict: &Dictionary, out: &mut Vec<u8>) -> Result<usize, Error> {
    once(input, Some(dict), out)
}

/// One-shot decoding into a new vector, as `once`, with the vector made and the result written by
/// `run_vec`.
#[inline(always)]
fn owned(input: &[u8], dict: Option<&Dictionary>) -> Result<Vec<u8>, Error> {
    let mut ctx = MaybeUninit::<Ctx>::uninit();
    Ctx::make(&mut ctx, input);
    let mut r = MaybeUninit::uninit();
    // SAFETY: as in `once`.
    unsafe {
        run_vec(&mut r, ctx.assume_init_mut(), dict);
        Ctx::free(&mut ctx);
        r.assume_init()
    }
}

/// One-shot decoding: the context lives in this frame, made (with the input bound) and dropped by
/// out-of-line calls, and `run` writes the result, so the caller holds only the output, the dictionary and
/// the result's address across the calls and no value with drop glue (a decoder there kept an unwind path
/// that dropped it, and its `Option` was tested and copied in the caller).
#[inline(always)]
fn once(input: &[u8], dict: Option<&Dictionary>, out: &mut Vec<u8>) -> Result<usize, Error> {
    let mut ctx = MaybeUninit::<Ctx>::uninit();
    Ctx::make(&mut ctx, input);
    let mut r = MaybeUninit::uninit();
    // SAFETY: `make` wrote a whole context; it is dropped once, after its last use. `run` writes `r`.
    unsafe {
        run(&mut r, ctx.assume_init_mut(), out, dict);
        Ctx::free(&mut ctx);
        r.assume_init()
    }
}

/// A reusable decoder. It keeps its working memory between calls, so decoding many frames with one
/// `Decoder` is faster than calling [`decompress`] each time.
pub struct Decoder {
    ctx: Option<Ctx>,
    cpu: Cpu,
}

impl Default for Decoder {
    fn default() -> Self {
        Decoder::new()
    }
}

impl Decoder {
    /// A new decoder, using the fastest code path this CPU supports.
    pub fn new() -> Self {
        Decoder::on(Cpu::detect())
    }

    fn on(cpu: Cpu) -> Self {
        Decoder { ctx: None, cpu }
    }

    /// Decompresses every frame in `input` into a new vector.
    pub fn decompress(&mut self, input: &[u8]) -> Result<Vec<u8>, Error> {
        self.guarded_vec(input, None)
    }

    /// Decompresses every frame in `input`, appending to `out`. Returns the number of bytes appended.
    pub fn decompress_into(&mut self, input: &[u8], out: &mut Vec<u8>) -> Result<usize, Error> {
        self.guarded(input, None, out)
    }

    /// [`Decoder::decompress`] for frames compressed with a dictionary.
    pub fn decompress_with_dict(&mut self, input: &[u8], dict: &Dictionary) -> Result<Vec<u8>, Error> {
        self.guarded_vec(input, Some(dict))
    }

    /// [`Decoder::decompress_into`] for frames compressed with a dictionary.
    pub fn decompress_into_with_dict(&mut self, input: &[u8], dict: &Dictionary, out: &mut Vec<u8>) -> Result<usize, Error> {
        self.guarded(input, Some(dict), out)
    }

    /// `run` with the context alone; the error is read from its job. The first call goes through `first`,
    /// which makes the context and comes back here: made inline, the arguments were held across that call.
    #[inline(always)]
    fn guarded(&mut self, input: &[u8], dict: Option<&Dictionary>, out: &mut Vec<u8>) -> Result<usize, Error> {
        let Some(ctx) = &mut self.ctx else {
            return self.first(input, dict, out);
        };
        bind(ctx, input);
        let mut r = MaybeUninit::uninit();
        run(&mut r, ctx, out, dict);
        // SAFETY: `run` wrote `r`.
        unsafe { r.assume_init() }
    }

    #[inline(never)]
    #[cold]
    fn first(&mut self, input: &[u8], dict: Option<&Dictionary>, out: &mut Vec<u8>) -> Result<usize, Error> {
        self.make();
        self.guarded(input, dict, out)
    }

    /// `guarded` into a new vector, made and written into the result by `run_vec`.
    #[inline(always)]
    fn guarded_vec(&mut self, input: &[u8], dict: Option<&Dictionary>) -> Result<Vec<u8>, Error> {
        let Some(ctx) = &mut self.ctx else {
            return self.first_vec(input, dict);
        };
        bind(ctx, input);
        let mut r = MaybeUninit::uninit();
        run_vec(&mut r, ctx, dict);
        // SAFETY: `run_vec` wrote `r`.
        unsafe { r.assume_init() }
    }

    #[inline(never)]
    #[cold]
    fn first_vec(&mut self, input: &[u8], dict: Option<&Dictionary>) -> Result<Vec<u8>, Error> {
        self.make();
        self.guarded_vec(input, dict)
    }

    #[inline(always)]
    fn make(&mut self) {
        // SAFETY: `ctx` is `None`, so nothing is dropped by overwriting it.
        unsafe { ptr::write(&mut self.ctx, Some(Ctx::new(self.cpu))) };
    }
}

/// The sum of the content sizes of the frames in `input`, `None` at the first frame without one.
pub fn content_size(input: &[u8]) -> Result<Option<u64>, Error> {
    match sizes(input) {
        (total, SIZES_ALL) => Ok(Some(total)),
        (_, SIZES_OPEN) => Ok(None),
        (_, SIZES_RESERVED) => Err(Error::ReservedBit),
        (_, SIZES_MAGIC) => Err(Error::BadMagic),
        (_, SIZES_BLOCK) => Err(Error::BlockType),
        (_, SIZES_OVERFLOW) => Err(Error::ContentSize),
        _ => Err(Error::Truncated),
    }
}

/// What `sizes` returns with its total: every frame sized, a frame without its size, or the error.
const SIZES_ALL: u32 = 0;
const SIZES_OPEN: u32 = 1;
const SIZES_TRUNCATED: u32 = 2;
const SIZES_RESERVED: u32 = 3;
const SIZES_MAGIC: u32 = 4;
const SIZES_BLOCK: u32 = 5;
const SIZES_OVERFLOW: u32 = 6;

/// `content_size` as its total and a `SIZES_*` code.
///
/// One walk over the input with a read pointer and the bytes left: the header fields it needs
/// (descriptor, content size) are read where they lie and the blocks are walked by `walk`, so nothing
/// goes through memory but the input. With the header written into a local and `header` called, the loop
/// kept eight callee-saved registers and stack slots. On Win64 the walk takes all seven volatile
/// registers (the result's address, the total, the read pointer, the bytes left, the checksum length and
/// two temporaries). Each exit returns its own code: with `Result` and `?` the errors met in one store of
/// a code chosen on the way, held in an eighth register (x86_64 8 moves, wasm32 358 instructions against
/// 304).
#[inline(always)]
fn sizes(input: &[u8]) -> (u64, u32) {
    if input.is_empty() {
        return (0, SIZES_TRUNCATED);
    }
    let mut total = 0u64;
    let (mut q, mut left) = (input.as_ptr(), input.len());
    // SAFETY: `left` bytes are readable at `q`; every read below is checked against `left` first.
    unsafe {
        while left != 0 {
            if left < 4 {
                return (0, SIZES_TRUNCATED);
            }
            let m = u32::from_le_bytes(q.cast::<[u8; 4]>().read());
            if m == MAGIC {
                if left == 4 {
                    return (0, SIZES_TRUNCATED);
                }
                let fhd = q.add(4).read();
                if fhd & 0x08 != 0 {
                    return (0, SIZES_RESERVED);
                }
                let n = 4 + head_len(q.add(4));
                if n > left {
                    return (0, SIZES_TRUNCATED);
                }
                // The descriptor read again, so the length's temporaries end at the check; then the read
                // pointer moves to the header's end, where the content size field ends.
                let fhd = ptr::read(q.add(4));
                q = q.add(n);
                left -= n;
                if fhd & 0x20 == 0 && fhd >> 6 == 0 {
                    return (0, SIZES_OPEN);
                }
                let Some(t) = total.checked_add(fcs_end(q, fhd)) else {
                    return (0, SIZES_OVERFLOW);
                };
                total = t;
                // The checksum flag is bit 2: 4 checksum bytes or none.
                match walk(&mut q, &mut left, (fhd & 0x04) as usize) {
                    Ok(()) => {}
                    Err(Error::BlockType) => return (0, SIZES_BLOCK),
                    Err(_) => return (0, SIZES_TRUNCATED),
                }
            } else if m & 0xFFFF_FFF0 == SKIP_MAGIC {
                if left < 8 {
                    return (0, SIZES_TRUNCATED);
                }
                let size = u32::from_le_bytes(q.add(4).cast::<[u8; 4]>().read()) as usize;
                if size > left - 8 {
                    return (0, SIZES_TRUNCATED);
                }
                q = q.add(8 + size);
                left -= 8 + size;
            } else {
                return (0, SIZES_MAGIC);
            }
        }
    }
    (total, SIZES_ALL)
}

/// The dictionary id of the first frame in `input` after any skippable frames.
///
/// Read the way `sizes` reads: a read pointer and the bytes left, the header checked whole and its id
/// read where it lies, so the function calls nothing (calling `header` it wrote the header into a stack
/// local and held its skip loop in callee-saved registers).
pub fn dictionary_id(input: &[u8]) -> Result<u32, Error> {
    let (mut q, mut left) = (input.as_ptr(), input.len());
    // SAFETY: `left` bytes are readable at `q`; every read below is checked against `left` first.
    unsafe {
        loop {
            if left < 4 {
                return Err(Error::Truncated);
            }
            let m = u32::from_le_bytes(q.cast::<[u8; 4]>().read());
            if m == MAGIC {
                if left == 4 {
                    return Err(Error::Truncated);
                }
                let fhd = q.add(4).read();
                if fhd & 0x08 != 0 {
                    return Err(Error::ReservedBit);
                }
                if 4 + head_len(q.add(4)) > left {
                    return Err(Error::Truncated);
                }
                // The id follows the descriptor and the window byte, if any; read as `header` reads it.
                let fhd = ptr::read(q.add(4));
                let d = q.add(6 - (fhd >> 5 & 1) as usize);
                return Ok(if fhd & 2 != 0 {
                    if fhd & 1 != 0 { u32::from_le_bytes(d.cast::<[u8; 4]>().read()) } else { u16::from_le_bytes(d.cast::<[u8; 2]>().read()) as u32 }
                } else if fhd & 1 != 0 {
                    d.read() as u32
                } else {
                    0
                });
            } else if m & 0xFFFF_FFF0 == SKIP_MAGIC {
                if left < 8 {
                    return Err(Error::Truncated);
                }
                let size = u32::from_le_bytes(q.add(4).cast::<[u8; 4]>().read()) as usize;
                if size > left - 8 {
                    return Err(Error::Truncated);
                }
                q = q.add(8 + size);
                left -= 8 + size;
            } else {
                return Err(Error::BadMagic);
            }
        }
    }
}

#[cfg(feature = "compat")]
pub(crate) fn content_size_first(input: &[u8]) -> Result<Option<u64>, Error> {
    match read_le(input, 0, 4)? as u32 {
        MAGIC => Ok(read_header(input, 4, &mut MaybeUninit::uninit())?.0.fcs()),
        m if m & 0xFFFF_FFF0 == SKIP_MAGIC => Ok(Some(0)),
        _ => Err(Error::BadMagic),
    }
}

#[cfg(feature = "compat")]
pub(crate) fn frame_extent(input: &[u8]) -> Result<Option<usize>, Error> {
    let end = match read_le(input, 0, 4) {
        Err(_) => return Ok(None),
        Ok(m) if m as u32 == MAGIC => read_header(input, 4, &mut MaybeUninit::uninit()).and_then(|(h, len)| skip_blocks(input, 4 + len, h.checksum)),
        Ok(m) if m as u32 & 0xFFFF_FFF0 == SKIP_MAGIC => skip_frame(input, 0),
        Ok(_) => return Err(Error::BadMagic),
    };
    match end {
        Ok(n) => Ok(Some(n)),
        Err(Error::Truncated) => Ok(None),
        Err(e) => Err(e),
    }
}

const SKIP_MAGIC: u32 = 0x184D_2A50;

#[cfg(any(test, feature = "compat"))]
fn read_le(input: &[u8], p: usize, n: usize) -> Result<u64, Error> {
    let b = input.get(p..p + n).ok_or(Error::Truncated)?;
    Ok(b.iter().rev().fold(0u64, |v, &x| (v << 8) | x as u64))
}

#[cfg(any(test, feature = "compat"))]
fn skip_frame(input: &[u8], p: usize) -> Result<usize, Error> {
    let size = read_le(input, p + 4, 4)? as usize;
    let end = (p + 8).checked_add(size).ok_or(Error::Truncated)?;
    if end > input.len() {
        return Err(Error::Truncated);
    }
    Ok(end)
}

/// The end of the frame whose first block starts at `p` (at most `input.len()`).
#[cfg(feature = "compat")]
fn skip_blocks(input: &[u8], p: usize, checksum: bool) -> Result<usize, Error> {
    let (mut q, mut left) = (input[p..].as_ptr(), input.len() - p);
    // SAFETY: `left` bytes are readable at `q`.
    unsafe { walk(&mut q, &mut left, 4 * checksum as usize)? };
    Ok(input.len() - left)
}

/// Moves the read pointer `q` (with `left` bytes after it) from the first block header of a frame to the
/// frame's end: each block header read where it lies and its block skipped, then the `ck` checksum bytes
/// (0 or 4) after the last block.
///
/// # Safety
/// `left` bytes are readable at `q`.
#[inline(always)]
unsafe fn walk(q: &mut *const u8, left: &mut usize, ck: usize) -> Result<(), Error> {
    loop {
        if *left < 3 {
            return Err(Error::Truncated);
        }
        // The block header's first byte (last-block bit, type, low size bits) is read where each test
        // needs it, so no part of it waits in a register.
        // SAFETY: three bytes are left.
        let kind = unsafe { ptr::read(*q) } & 6;
        if kind == 6 {
            return Err(Error::BlockType);
        }
        // An RLE block holds one byte.
        let size = 3 + if kind == 2 {
            1
        } else {
            // SAFETY: as above.
            unsafe { (u16::from_le_bytes(q.cast::<[u8; 2]>().read()) as usize | (q.add(2).read() as usize) << 16) >> 3 }
        };
        if size > *left {
            return Err(Error::Truncated);
        }
        // SAFETY: as above; then `size <= left`.
        let last = unsafe { ptr::read(*q) } & 1 != 0;
        *q = unsafe { q.add(size) };
        *left -= size;
        if last {
            break;
        }
    }
    if ck > *left {
        return Err(Error::Truncated);
    }
    // SAFETY: `ck <= left`.
    *q = unsafe { q.add(ck) };
    *left -= ck;
    Ok(())
}

/// The caller's part of `run`'s binding: the input, stored through the context the caller holds anyway,
/// so `run` takes four register arguments.
#[inline(always)]
fn bind(ctx: &mut Ctx, input: &[u8]) {
    (ctx.job.inp, ctx.job.inp_len) = (input.as_ptr() as usize, input.len());
}

/// Decodes the frames of the input bound in the job onto the end of `out` and writes the result to `r`:
/// the bytes written, or the error with `out` cut back to its length at the call. The output is bound in
/// the job too, so the frame loop holds the context, the dictionary and the output's length at the call
/// across its calls; the result is written here, so the caller holds nothing for it (returned as a count,
/// the caller held the context for the error). The dictionary and the length are not context fields: the
/// stream decoder's fields after its context moved with them, and its 1-byte feeds were 13% slower.
#[inline(never)]
fn run(r: &mut MaybeUninit<Result<usize, Error>>, ctx: &mut Ctx, out: &mut Vec<u8>, dict: Option<&Dictionary>) {
    let start = out.len();
    (ctx.job.ip, ctx.job.out) = (0, ptr::from_mut(out) as usize);
    if frames(ctx, dict) {
        // SAFETY: `vec` is the bound output vector.
        r.write(Ok(unsafe { (*vec(ctx)).len() } - start));
        return;
    }
    // SAFETY: as above.
    unsafe { (*vec(ctx)).truncate(start) };
    r.write(Err(at(&ctx.job.err).0));
}

/// `run` into a new vector made in the result itself, so neither this function nor its caller copies it
/// (on the caller's stack and moved into the result, its fields were stored, read back and stored again).
#[inline(never)]
fn run_vec(r: &mut MaybeUninit<Result<Vec<u8>, Error>>, ctx: &mut Ctx, dict: Option<&Dictionary>) {
    let Ok(out) = r.write(Ok(Vec::new())) else { unreachable!() };
    (ctx.job.ip, ctx.job.out) = (0, ptr::from_mut(out) as usize);
    if !frames(ctx, dict) {
        // SAFETY: `r` holds the vector; it is dropped once and `r` is written again.
        unsafe { r.assume_init_drop() };
        r.write(Err(at(&ctx.job.err).0));
    }
}

/// The frames of the input bound in the job, from `ip` to its end; false with the error in the job. The
/// position lives in the job, and every value is read from there where used.
#[inline(always)]
fn frames(ctx: &mut Ctx, dict: Option<&Dictionary>) -> bool {
    if at(&ctx.job.inp_len) == 0 {
        return ctx.truncated();
    }
    loop {
        let (inp, p, len) = (at(&ctx.job.inp) as *const u8, at(&ctx.job.ip), at(&ctx.job.inp_len));
        if p >= len {
            return true;
        }
        if len - p < 4 {
            return ctx.truncated();
        }
        // SAFETY: `p + 4 <= len`; the input is live for the call.
        let m = unsafe { u32::from_le_bytes(inp.add(p).cast::<[u8; 4]>().read()) };
        if m == MAGIC {
            put(&mut ctx.job.ip, p + 4);
            if !frame(ctx, dict) {
                return false;
            }
        } else if m & 0xFFFF_FFF0 == SKIP_MAGIC {
            if len - p < 8 {
                return ctx.truncated();
            }
            // SAFETY: `p + 8 <= len`.
            let size = unsafe { u32::from_le_bytes(inp.add(p + 4).cast::<[u8; 4]>().read()) } as usize;
            match (p + 8).checked_add(size) {
                Some(end) if end <= len => put(&mut ctx.job.ip, end),
                _ => return ctx.truncated(),
            }
        } else {
            return ctx.fault(Error::BadMagic);
        }
    }
}

/// A frame header: its content size (0 unless `sized`), window, dictionary id and whether it has a
/// checksum. Written by `header` field by field.
#[derive(Clone, Copy)]
struct Header {
    fcs: u64,
    window: u64,
    did: u32,
    sized: bool,
    checksum: bool,
}

impl Header {
    fn fcs(&self) -> Option<u64> {
        self.sized.then_some(self.fcs)
    }
}

/// What `header` returns for a short input and for the reserved bit; every header is at least 2 bytes.
const HEAD_TRUNCATED: usize = 0;
const HEAD_RESERVED: usize = 1;

/// Reads the frame header at the start of `src` (its descriptor byte) into `h`: its length with every
/// field written, or `HEAD_TRUNCATED` / `HEAD_RESERVED`. The field sizes come from the descriptor
/// (dictionary id 0, 1, 2 or 4 bytes; content size 1 or 0 by the single-segment flag, else 2, 4 or 8),
/// so the whole header is bounds-checked once, each field is one load of its width, and each is stored
/// when known so that nothing waits in a register for the rest.
///
/// The caller adds the length to its own position: with the position an argument here, it was the
/// eighth value at the bounds check on x86. The length is a register result, known at the bounds check,
/// which a caller that walks on to the blocks waits for (stored in `h`, or as the address after the
/// fields, the walk measured slower).
///
/// Its own function: since `content_size` and `dictionary_id` read their fields in place, the frame loop
/// is its only caller and inlined it, which changed the frame loop's register allocation; with every
/// function on a 4 KB boundary that lost the 1 KB-block file 0.5% and a reused decoder's frame 1.3%
/// (out of line: 0.8% faster, and a tie). It keeps its own length arithmetic and field reads: with
/// `head_len` (the descriptor re-read per part) a reused decoder's frame took 1.6% longer, and with the
/// content size read through one reader shared with `content_size` it was 3 instructions longer.
#[inline(never)]
fn header(src: &[u8], h: &mut MaybeUninit<Header>) -> usize {
    let h = h.as_mut_ptr();
    let Some(&fhd) = src.first() else {
        return HEAD_TRUNCATED;
    };
    if fhd & 0x08 != 0 {
        return HEAD_RESERVED;
    }
    // SAFETY (each field access below): `h` is a valid, writable `Header`.
    unsafe { ptr::write(&raw mut (*h).checksum, fhd & 0x04 != 0) };
    let single = fhd & 0x20 != 0;
    let (d, code) = ((fhd & 3) as usize, (fhd >> 6) as usize);
    // The window byte and the content size together: 2 for code 0 (the window byte, or the 1-byte
    // size), else the window byte or nothing plus 2, 4 or 8 bytes. The dictionary id: 0, 1, 2 or 4.
    // No shifts: on x86 without BMI2 a variable shift takes `cl`.
    let did_size = d + (d == 3) as usize;
    let rest = if code == 0 { 2 } else { 2 - single as usize + 2 * (code + (code == 3) as usize) };
    let n = did_size + rest;
    if n > src.len() {
        return HEAD_TRUNCATED;
    }
    // The fields in order, each branch advancing by its constant width. The descriptor is read again,
    // so the sizes above end at the bounds check instead of waiting in registers.
    // SAFETY: every read below lies in `src[..n]`; `h` as above.
    unsafe {
        let mut q = src.as_ptr();
        let fhd = ptr::read(q);
        let single = fhd & 0x20 != 0;
        q = q.add(1);
        if !single {
            let wd = q.read();
            q = q.add(1);
            // 2^(10 + e) + 2^(7 + e) * m as one shift: one temporary fewer than the base and its eighth.
            ptr::write(&raw mut (*h).window, (8 + (wd & 7) as u64) << (7 + (wd >> 3) as u32));
        }
        // Bit tests, not a `match`: a switch of four cases is a jump table, an indirect jump per field.
        // (`dictionary_id` reads the id the same way; one reader for both, with the width returned or
        // computed apart, cost this function 5 instructions on x86_64 and 4 to 16 on aarch64.)
        let mut did = 0;
        if fhd & 2 != 0 {
            if fhd & 1 != 0 {
                did = u32::from_le_bytes(q.cast::<[u8; 4]>().read());
                q = q.add(4);
            } else {
                did = u16::from_le_bytes(q.cast::<[u8; 2]>().read()) as u32;
                q = q.add(2);
            }
        } else if fhd & 1 != 0 {
            did = q.read() as u32;
            q = q.add(1);
        }
        ptr::write(&raw mut (*h).did, did);
        let fcs = if fhd & 0x80 != 0 {
            if fhd & 0x40 != 0 { u64::from_le_bytes(q.cast::<[u8; 8]>().read()) } else { u32::from_le_bytes(q.cast::<[u8; 4]>().read()) as u64 }
        } else if fhd & 0x40 != 0 {
            // A 2-byte content size counts from 256.
            u16::from_le_bytes(q.cast::<[u8; 2]>().read()) as u64 + 256
        } else if single {
            q.read() as u64
        } else {
            0
        };
        ptr::write(&raw mut (*h).fcs, fcs);
        ptr::write(&raw mut (*h).sized, single || fhd >> 6 != 0);
        if single {
            ptr::write(&raw mut (*h).window, fcs);
        }
    }
    n
}

/// 0, 1, 2, 4 for a two-bit field 0, 1, 2, 3 in one register: `x + (x == 3)` needs a second for the
/// compare.
#[inline(always)]
fn pow_field(x: usize) -> usize {
    (5 * x + 1) >> 2
}

/// The length of the frame header by its descriptor at `d`, the descriptor included. The window byte and
/// the content size together: 2 for code 0 (the window byte, or the 1-byte size), else the window byte
/// or nothing plus 2, 4 or 8 bytes; the dictionary id 0, 1, 2 or 4. No shifts: on x86 without BMI2 a
/// variable shift takes `cl`. The descriptor is read where each part needs it, so no copy of it waits
/// beside the parts.
///
/// # Safety
/// `d` is readable.
#[inline(always)]
unsafe fn head_len(d: *const u8) -> usize {
    // SAFETY: by the caller.
    let fhd = || unsafe { ptr::read(d) };
    let n = pow_field((fhd() & 3) as usize) + 2;
    let code = (fhd() >> 6) as usize;
    if code == 0 { n } else { n + 2 * pow_field(code) - (fhd() >> 5 & 1) as usize }
}

/// The content size field by the descriptor, read back from `end`, the end of the header (the field is its
/// last, so no field offset is computed): 8, 4, 2 (counting from 256) or 1 byte (single segment), else 0,
/// each width one load at a constant offset.
///
/// # Safety
/// The header the descriptor describes ends at `end`, readable.
#[inline(always)]
unsafe fn fcs_end(end: *const u8, fhd: u8) -> u64 {
    // SAFETY: by the caller. Bit tests, not a `match`: a switch of four cases is a jump table.
    unsafe {
        if fhd & 0x80 != 0 {
            if fhd & 0x40 != 0 {
                u64::from_le_bytes(end.sub(8).cast::<[u8; 8]>().read())
            } else {
                u32::from_le_bytes(end.sub(4).cast::<[u8; 4]>().read()) as u64
            }
        } else if fhd & 0x40 != 0 {
            u16::from_le_bytes(end.sub(2).cast::<[u8; 2]>().read()) as u64 + 256
        } else if fhd & 0x20 != 0 {
            end.sub(1).read() as u64
        } else {
            0
        }
    }
}

/// `header` of the frame header at `p`, with its codes as an `Error` for `?`: the header in `h` (with no
/// zeroing before) and its length; its first block starts at `p + len`.
#[cfg(any(test, feature = "compat"))]
#[inline(always)]
fn read_header<'a>(input: &[u8], p: usize, h: &'a mut MaybeUninit<Header>) -> Result<(&'a Header, usize), Error> {
    let src = input.get(p..).ok_or(Error::Truncated)?;
    match header(src, h) {
        HEAD_TRUNCATED => Err(Error::Truncated),
        HEAD_RESERVED => Err(Error::ReservedBit),
        // SAFETY: `header` wrote every field when it returned a length.
        n => Ok((unsafe { h.assume_init_ref() }, n)),
    }
}

/// The frame whose header starts at the job's `ip` (after the magic number); false with the error in the
/// job. The input and the output come from the job, `ip` is left after the frame, and every header field
/// is stored in the job as soon as it is read, and read back from there after each call, so nothing is held
/// across the header reader, the reset or the reserve; the dictionary waits in the job's history field,
/// which takes the history after the reset. Its own function: inlined into the frames loop, the hasher's
/// constants and the header's address were hoisted into callee-saved registers.
#[inline(never)]
fn frame(ctx: &mut Ctx, dict: Option<&Dictionary>) -> bool {
    put(&mut ctx.job.ext, dict.map_or(0, |d| ptr::from_ref(d) as usize));
    let (inp, p) = (at(&ctx.job.inp) as *const u8, at(&ctx.job.ip));
    // SAFETY: `p <= inp_len` (the caller read the magic number before it); the input is live for the call.
    let src = unsafe { core::slice::from_raw_parts(inp.add(p), at(&ctx.job.inp_len) - p) };
    let len = header(src, &mut ctx.head);
    if len == HEAD_TRUNCATED {
        return ctx.fail(Error::Truncated);
    }
    if len == HEAD_RESERVED {
        return ctx.fail(Error::ReservedBit);
    }
    let q = at(&ctx.job.ip) + len;
    put(&mut ctx.job.ip, q);
    {
        // SAFETY: `header` wrote every field when it returned a length.
        let h = unsafe { ctx.head.assume_init_ref() };
        let (fcs, sized, check, window) = (at(&h.fcs), at(&h.sized), at(&h.checksum), at(&h.window));
        let j = &mut ctx.job;
        (j.fcs, j.sized, j.check) = (fcs, sized, check);
        (j.window, j.block_max, j.ring) = (usize::try_from(window).unwrap_or(usize::MAX), window.min(BLOCK_MAX as u64) as usize, false);
    }
    // SAFETY: as above.
    let did = at(&unsafe { ctx.head.assume_init_ref() }.did);
    // SAFETY: stored above from a live dictionary or 0, live for the call.
    let dict = unsafe { (at(&ctx.job.ext) as *const Dictionary).as_ref() };
    if did != 0 && dict.is_none_or(|d| d.id != did) {
        return ctx.fail(Error::Dictionary(did));
    }
    reset(ctx, dict);
    // SAFETY: as above.
    let history: &[u8] = unsafe { (at(&ctx.job.ext) as *const Dictionary).as_ref() }.map_or(&[], |d| &d.content);
    (ctx.job.ext, ctx.job.ext_len) = (history.as_ptr() as usize, history.len());
    // SAFETY: `vec` is the bound output vector.
    let frame_start = unsafe { (*vec(ctx)).len() };
    (ctx.job.fs, ctx.job.hashed) = (frame_start, frame_start);
    if at(&ctx.job.sized) {
        let cap = (at(&ctx.job.inp_len) as u64).saturating_mul(1024).max(1 << 20);
        let n = at(&ctx.job.fcs).min(cap) as usize + at(&ctx.job.block_max) + SLACK;
        // SAFETY: as above.
        unsafe { (*vec(ctx)).reserve(n) };
    }
    if at(&ctx.job.check) {
        ctx.sum = Xxh64::new(0);
    }
    if !blocks(ctx) {
        return false;
    }
    // SAFETY: as above.
    let out = unsafe { &*vec(ctx) };
    if at(&ctx.job.sized) && (out.len() - at(&ctx.job.fs)) as u64 != at(&ctx.job.fcs) {
        return ctx.fail(Error::ContentSize);
    }
    if at(&ctx.job.check) {
        if at(&ctx.job.inp_len) - at(&ctx.job.ip) < 4 {
            return ctx.fail(Error::Truncated);
        }
        // The stored sum and the position are read after the hash, so neither waits through it.
        let sum = ctx.sum.finish_with(&out[at(&ctx.job.hashed)..]) as u32;
        let p = at(&ctx.job.ip);
        // SAFETY: `p + 4 <= inp_len`, checked above.
        if sum != unsafe { u32::from_le_bytes((at(&ctx.job.inp) as *const u8).add(p).cast::<[u8; 4]>().read()) } {
            return ctx.fail(Error::Checksum);
        }
        put(&mut ctx.job.ip, p + 4);
    }
    true
}

/// `Ctx::reset` for `frame`, out of line as the bench program's frame code always had it (inlined into
/// `frame`, 256 small frames through the stream decoder took 12% longer in default layout). The stream
/// decoder's `begin` calls `Ctx::reset` itself, and a program inlines it there as before: out of line for
/// both, the 256-small-frames timing was 2% to 8% slower.
#[inline(never)]
fn reset(ctx: &mut Ctx, dict: Option<&Dictionary>) {
    ctx.reset(dict);
}

/// The blocks of the frame bound in the job, through the last; false with the error in the job. Every
/// value lives in the job (each read where used, each result stored when known), so the loop holds the
/// context alone across the copies, the fills, the block decoder and the checksum. The checksum is
/// taken over the output where it lies, whole stripes after each block, the tail at the frame's end.
#[inline]
fn blocks(ctx: &mut Ctx) -> bool {
    loop {
        let (inp, p) = (at(&ctx.job.inp) as *const u8, at(&ctx.job.ip));
        let left = at(&ctx.job.inp_len) - p;
        // The failures go through the cold leaves: inline, their stores merged and the truncation's tag
        // waited in a callee-saved register through the loop.
        if left < 3 {
            return ctx.truncated();
        }
        // SAFETY: `p + 3 <= inp_len`; the input is live for the frame.
        let bh = unsafe { inp.add(p).read() as usize | (inp.add(p + 1).read() as usize) << 8 | (inp.add(p + 2).read() as usize) << 16 };
        let bsize = bh >> 3;
        if bsize > at(&ctx.job.block_max) {
            return ctx.fault(Error::BlockSize);
        }
        put(&mut ctx.job.last, bh & 1 != 0);
        let kind = (bh >> 1) & 3;
        if kind == 3 {
            return ctx.fault(Error::BlockType);
        }
        let need = if kind == 1 { 1 } else { bsize };
        if need > left - 3 {
            return ctx.truncated();
        }
        // The block body and its decoded size (a fill's run, a raw or compressed block's size), read back
        // from the job after the reserve, so the loop holds nothing across that call either.
        put(&mut ctx.job.ip, p + 3 + need);
        // SAFETY: `p + 3 + need <= inp_len`.
        put(&mut ctx.job.blk, unsafe { inp.add(p + 3) } as usize);
        put(&mut ctx.job.len, bsize);
        if kind == 0 {
            // The destination first: a source read before `grow` would be held across its reserve call.
            let dst = grow(ctx);
            // SAFETY: `grow` made room for `len` bytes at `dst`; the body has `len` bytes.
            unsafe { ptr::copy_nonoverlapping(at(&ctx.job.blk) as *const u8, dst, at(&ctx.job.len)) };
        } else if kind == 1 {
            let dst = grow(ctx);
            // SAFETY: as above; the body is the fill byte.
            unsafe { ptr::write_bytes(dst, (at(&ctx.job.blk) as *const u8).read(), at(&ctx.job.len)) };
        } else {
            // SAFETY: `vec` is the bound output vector.
            unsafe {
                (*vec(ctx)).reserve(at(&ctx.job.block_max) + SLACK);
                let out = &mut *vec(ctx);
                let j = &mut ctx.job;
                (j.base, j.pos, j.cap) = (out.as_mut_ptr() as usize, out.len(), out.capacity());
            }
            if !compressed(ctx) {
                return false;
            }
            // SAFETY: `compressed` wrote the output up to `pos <= cap`.
            unsafe { (*vec(ctx)).set_len(at(&ctx.job.pos)) };
        }
        // SAFETY: as above.
        if at(&ctx.job.sized) && (unsafe { (*vec(ctx)).len() } - at(&ctx.job.fs)) as u64 > at(&ctx.job.fcs) {
            return ctx.fault(Error::ContentSize);
        }
        if at(&ctx.job.check) {
            checksum(ctx);
        }
        if at(&ctx.job.last) {
            return true;
        }
    }
}

/// The output vector `frame` bound in the job (valid for the frame, referred to by nothing else
/// meanwhile), its address read where used.
#[inline(always)]
fn vec(ctx: &Ctx) -> *mut Vec<u8> {
    at(&ctx.job.out) as *mut Vec<u8>
}

/// Makes room for the job's `len` bytes at the end of the output, sets the length past them and returns
/// their address: the caller's copy or fill writes them before anything reads them.
#[inline(always)]
fn grow(ctx: &mut Ctx) -> *mut u8 {
    // SAFETY: `vec` is the bound output vector; `n` bytes are reserved at `end`.
    unsafe {
        (*vec(ctx)).reserve(at(&ctx.job.len));
        let (out, n) = (&mut *vec(ctx), at(&ctx.job.len));
        let end = out.len();
        out.set_len(end + n);
        out.as_mut_ptr().add(end)
    }
}

/// Hashes the whole stripes of the output past `hashed`. A leaf taking the context alone, so the block
/// loop holds nothing across it and the hash's primes are not held in the block loop's registers.
#[inline(never)]
fn checksum(ctx: &mut Ctx) {
    let (h, out) = (ctx.job.hashed, ctx.job.out as *const Vec<u8>);
    // SAFETY: `frame` bound the output vector; `hashed` starts at the frame start and grows by the
    // bytes hashed, so it never passes the length.
    let rest = unsafe { core::slice::from_raw_parts((*out).as_ptr().add(h), (*out).len() - h) };
    // Stored first, so the index does not wait in a register through the stripes.
    put(&mut ctx.job.hashed, h + (rest.len() & !31));
    ctx.sum.absorb(rest);
}

/// Literals section kinds `section` reports: raw read in place, raw to copy, a fill, Huffman with its
/// tree, Huffman with the previous tree; for `LIT_BAD` the error is in the job.
const LIT_HERE: u32 = 0;
const LIT_COPY: u32 = 1;
const LIT_FILL: u32 = 2;
const LIT_TREE: u32 = 3;
const LIT_REPEAT: u32 = 4;
const LIT_BAD: u32 = 5;

/// Reads the literals section of the bound block; leaves the literals' address and count and the bytes
/// the section takes in the job. False with the error in the job.
#[inline(always)]
fn literals(ctx: &mut Ctx) -> bool {
    let k = section(ctx);
    if k == LIT_BAD {
        return false;
    }
    // SAFETY: `Ctx::lits` holds more than `lit_len <= block_max` bytes; a copy's source lies in the block.
    unsafe {
        if k == LIT_COPY {
            ptr::copy_nonoverlapping(at(&ctx.job.src) as *const u8, at(&ctx.job.lit) as *mut u8, at(&ctx.job.lit_len));
        } else if k == LIT_FILL {
            ptr::write_bytes(at(&ctx.job.lit) as *mut u8, at(&ctx.job.src) as u8, at(&ctx.job.lit_len));
        }
    }
    if k < LIT_TREE {
        return true;
    }
    match huffman(ctx, k == LIT_TREE) {
        Ok(()) => true,
        Err(e) => ctx.fail(e),
    }
}

/// The Huffman literals of the section `section` read: the tree (when the section has one), then the
/// streams into `Ctx::lits`. Every input is read from the job after the call before it.
#[inline(always)]
fn huffman(ctx: &mut Ctx, tree: bool) -> Result<(), Error> {
    let q = if tree {
        // SAFETY: `src .. src + src_len` is the Huffman section inside the bound block.
        let data = unsafe { core::slice::from_raw_parts(at(&ctx.job.src) as *const u8, at(&ctx.job.src_len)) };
        if !huf::tree(data, ctx.huf()) {
            return Err(ctx.huf().tree_error());
        }
        ctx.huf_valid = true;
        ctx.huf().tree_used()
    } else {
        0
    };
    let (csize, regen) = (at(&ctx.job.src_len), at(&ctx.job.lit_len));
    if q > csize {
        return Err(Error::Huffman);
    }
    // SAFETY: `src .. src + csize` is the Huffman section inside the bound block and `lit` is
    // `Ctx::lits`, which holds more than `regen <= block_max` bytes.
    let (streams, dst) =
        unsafe { (core::slice::from_raw_parts((at(&ctx.job.src) + q) as *const u8, csize - q), core::slice::from_raw_parts_mut(at(&ctx.job.lit) as *mut u8, regen)) };
    let floor = at(&ctx.job.blk) as *const u8;
    ctx.huf().bind(floor, streams, dst);
    // SAFETY: the table was just bound to `streams` inside the block and to `Ctx::lits`, all alive and unaliased for the call.
    unsafe {
        if at(&ctx.job.four) {
            huf::decode4(at(&ctx.cpu), ctx.huf())
        } else {
            huf::decode1(at(&ctx.cpu), ctx.huf())
        }
    }
}

/// Reads the literals section header of the bound block into the job: the literals' address and count,
/// the bytes the section takes, the source of the caller's next step (the raw literals to copy, the
/// fill byte or the Huffman section), and returns that step (`LIT_*`). A leaf: the copy, the fill and
/// the Huffman decode are the caller's calls.
#[inline(never)]
fn section(ctx: &mut Ctx) -> u32 {
    // Each failure stores its error as an immediate (a `Result` kept its tag in a register).
    macro_rules! fail {
        ($e:expr) => {{
            ctx.job.err = Fault($e);
            return LIT_BAD;
        }};
    }
    let len = ctx.job.len;
    // SAFETY: the caller of `compressed` bound the job to a live block for this call.
    let blk = unsafe { core::slice::from_raw_parts(ctx.job.blk as *const u8, len) };
    let Some(&b0) = blk.first() else { fail!(Error::Truncated) };
    let b0 = b0 as usize;
    // Header bytes past the section's own read as 0; no size field reaches them.
    let byte = |i: usize| blk.get(i).map_or(0, |&b| b as usize);
    let (ltype, sf) = (b0 & 3, (b0 >> 2) & 3);
    if ltype < 2 {
        // 1, 2 or 3 header bytes: a 5, 12 or 20-bit size.
        let (regen, h) = match sf {
            1 => ((b0 >> 4) + (byte(1) << 4), 2),
            3 => ((b0 >> 4) + (byte(1) << 4) + (byte(2) << 12), 3),
            _ => (b0 >> 3, 1),
        };
        if h > len {
            fail!(Error::Truncated);
        }
        if regen > at(&ctx.job.block_max) {
            fail!(Error::Literals);
        }
        // The length is read from the job where compared, so it is not held with the sizes.
        if ltype == 0 {
            if regen > at(&ctx.job.len) - h {
                fail!(Error::Truncated);
            }
            let d = blk.as_ptr() as usize + h;
            (ctx.job.lit_len, ctx.job.used, ctx.job.src) = (regen, h + regen, d);
            // Read in place when the copies' reads past the last literal stay inside the block.
            if at(&ctx.job.len) - (h + regen) >= SLACK {
                ctx.job.lit = d;
                LIT_HERE
            } else {
                ctx.job.lit = ctx.lits.as_ptr() as usize;
                LIT_COPY
            }
        } else {
            if h >= at(&ctx.job.len) {
                fail!(Error::Truncated);
            }
            (ctx.job.lit, ctx.job.lit_len, ctx.job.used, ctx.job.src) = (ctx.lits.as_ptr() as usize, regen, h + 1, byte(h));
            LIT_FILL
        }
    } else {
        // 3, 3, 4 or 5 header bytes: two sizes of 10, 10, 14 or 18 bits after the 4 type bits.
        // Each field is stored once known and the limits are read where compared, so the header word,
        // its size mask and the shift count fit the volatile registers with the block.
        let h = 3 + (sf >> 1) + (sf & (sf >> 1));
        ctx.job.four = sf != 0;
        if h > len {
            fail!(Error::Truncated);
        }
        let v = [1, 2, 3, 4].into_iter().fold(b0 as u64, |v, i| v | (byte(i) as u64) << (8 * i));
        put(&mut ctx.job.src, blk.as_ptr() as usize + h);
        let width = 4 * h as u32 - 2;
        let mask = (1u64 << width) - 1;
        let regen = ((v >> 4) & mask) as usize;
        if regen > at(&ctx.job.block_max) {
            fail!(Error::Literals);
        }
        put(&mut ctx.job.lit_len, regen);
        ctx.job.lit = ctx.lits.as_ptr() as usize;
        let csize = ((v >> (width + 4)) & mask) as usize;
        if csize > at(&ctx.job.len) - h {
            fail!(Error::Truncated);
        }
        (ctx.job.used, ctx.job.src_len) = (h + csize, csize);
        // The type's low bit: 0 for a section with its tree, 1 for one using the previous tree.
        if v & 1 == 0 {
            LIT_TREE
        } else if ctx.huf_valid {
            LIT_REPEAT
        } else {
            fail!(Error::Huffman)
        }
    }
}

/// A work block field read from memory where it is used: the compiler knows the calls in between do
/// not change it and would hold it in a register across them instead (a stack slot on Win64).
#[inline(always)]
pub(crate) fn at<T: Copy>(f: &T) -> T {
    // SAFETY: `f` is a valid reference.
    unsafe { ptr::read(f) }
}

/// A work block field written where it is computed: a plain store may sink past later work and keep
/// the value in a register until then.
#[inline(always)]
pub(crate) fn put<T: Copy>(f: &mut T, v: T) {
    // SAFETY: `f` is a valid reference.
    unsafe { ptr::write(f, v) }
}

/// The block bound in the job; false with the error left in the job. Every value lives in the job, the
/// tables or the sequence state, so nothing but the context is held across the calls. Inlinable: a
/// program with one caller inlines it into its frame loop (kept out of line, the call per block cost
/// w10 time).
#[inline]
pub(crate) fn compressed(ctx: &mut Ctx) -> bool {
    let j = &ctx.job;
    if j.pos > j.cap || j.cap - j.pos < j.block_max + SLACK || j.fs > j.pos {
        return ctx.fail(Error::BlockSize);
    }
    if !literals(ctx) {
        return false;
    }
    let (used, len) = (at(&ctx.job.used), at(&ctx.job.len));
    if used > len {
        return ctx.fail(Error::Truncated);
    }
    // SAFETY: the caller bound the job to a live block; `used <= len`.
    let rest = unsafe { core::slice::from_raw_parts((at(&ctx.job.blk) + used) as *const u8, len - used) };
    if !seq::parse_header(ctx.seq(), rest) {
        let e = ctx.seq().error();
        return ctx.fail(e);
    }
    if ctx.seq().count() == 0 {
        if !ctx.seq().stream().is_empty() {
            return ctx.fail(Error::Sequences);
        }
        let (base, lit, lit_len, pos) = (at(&ctx.job.base) as *mut u8, at(&ctx.job.lit) as *const u8, at(&ctx.job.lit_len), at(&ctx.job.pos));
        ctx.job.pos = pos + lit_len;
        // SAFETY: `compressed` checked `block_max + SLACK` bytes of room at `pos`; `lit_len <= block_max`.
        unsafe { ptr::copy_nonoverlapping(lit, base.add(pos), lit_len) };
        return true;
    }
    // SAFETY: checked on entry: `fs <= pos`, `pos + block_max + SLACK <= cap`; `lit_len + SLACK` bytes are
    // readable at `lit` (in the block, or `Ctx::lits` holds `BLOCK_MAX + 2 * SLACK`); the block, the
    // output and the history are live for the call.
    let r = unsafe {
        let cpu = at(&ctx.cpu);
        seq::bind(&mut ctx.seq, &ctx.job, cpu);
        let st = ctx.seq();
        let stream = st.stream();
        seq::execute(st, stream)
    };
    match r {
        Ok(()) => {
            let base = at(&ctx.job.base) as *mut u8;
            ctx.job.pos = ctx.seq().out(base);
            true
        }
        Err(e) => ctx.fail(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_with(blocks: &[u8], fcs: Option<u8>) -> Vec<u8> {
        let mut v = MAGIC.to_le_bytes().to_vec();
        match fcs {
            Some(n) => {
                v.push(0x20);
                v.push(n);
            }
            None => {
                v.push(0x00);
                v.push(0x00);
            }
        }
        v.extend_from_slice(blocks);
        v
    }

    #[test]
    fn raw_and_rle_blocks() {
        let blocks = [(3 << 3) as u8, 0, 0, b'a', b'b', b'c', ((5 << 3) | 2 | 1) as u8, 0, 0, b'z'];
        let out = decompress(&frame_with(&blocks, Some(8))).unwrap();
        assert_eq!(out, b"abczzzzz");
        assert_eq!(decompress(&frame_with(&blocks, Some(9))), Err(Error::ContentSize));
        assert_eq!(decompress(&frame_with(&blocks, None)).unwrap(), b"abczzzzz");
    }

    #[test]
    fn compressed_block_literals_only() {
        let blk = [(4 << 3) as u8, b'w', b'x', b'y', b'z', 0];
        let mut blocks = vec![((blk.len() << 3) | 4 | 1) as u8, 0, 0];
        blocks.extend_from_slice(&blk);
        assert_eq!(decompress(&frame_with(&blocks, None)).unwrap(), b"wxyz");
        assert_eq!(decompress(&frame_with(&blocks, Some(4))), Err(Error::BlockSize));
    }

    #[test]
    fn skippable_and_concatenated() {
        let blocks = [((2 << 3) | 1) as u8, 0, 0, b'h', b'i'];
        let f = frame_with(&blocks, Some(2));
        let mut input = f.clone();
        input.extend_from_slice(&[0x5A, 0x2A, 0x4D, 0x18, 3, 0, 0, 0, 9, 9, 9]);
        input.extend_from_slice(&f);
        assert_eq!(decompress(&input).unwrap(), b"hihi");
        let mut out = b"pre".to_vec();
        assert_eq!(decompress_into(&input, &mut out), Ok(4));
        assert_eq!(out, b"prehihi");
    }

    #[test]
    fn errors_not_panics() {
        assert_eq!(decompress(&[]), Err(Error::Truncated));
        assert_eq!(decompress(&[1, 2, 3, 4]), Err(Error::BadMagic));
        assert_eq!(decompress(&MAGIC.to_le_bytes()), Err(Error::Truncated));
        let mut d = MAGIC.to_le_bytes().to_vec();
        d.extend_from_slice(&[0x01, 0x00, 7]);
        assert_eq!(decompress(&d), Err(Error::Dictionary(7)));
        let mut r = MAGIC.to_le_bytes().to_vec();
        r.extend_from_slice(&[0x08, 0x00]);
        assert_eq!(decompress(&r), Err(Error::ReservedBit));
        let blocks = [((1 << 3) | 6 | 1) as u8, 0, 0, 0];
        assert_eq!(decompress(&frame_with(&blocks, None)), Err(Error::BlockType));
        let mut out = b"keep".to_vec();
        assert!(decompress_into(&frame_with(&blocks, None), &mut out).is_err());
        assert_eq!(out, b"keep");
    }

    mod embed {
        include!("../tests/data/embed.rs");
    }

    fn frames() -> Vec<(alloc::string::String, Vec<u8>, Option<Dictionary>)> {
        extern crate std;
        use alloc::string::ToString;
        let mut v = Vec::new();
        for line in embed::MANIFEST.lines().filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split_whitespace().collect();
            let dict = f.get(3).and_then(|x| x.strip_prefix("dict=")).map(|d| Dictionary::new(embed::file(d)).unwrap());
            v.push((f[0].to_string(), embed::file(f[0]).to_vec(), dict));
        }
        if let Some(extra) = std::env::var_os("ZYZYR_CORPUS") {
            if let Ok(rd) = std::fs::read_dir(extra) {
                for p in rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "zst")) {
                    if let Ok(data) = std::fs::read(&p) {
                        v.push((p.display().to_string(), data, None));
                    }
                }
            }
        }
        v
    }

    /// What `reference_dict` reads: the id, the content and the tables.
    type DictParts = (u32, Vec<u8>, Box<HufTable>, Box<SeqState>);

    /// The dictionary reader `Dictionary::make` replaced: the tree first, then the three tables, the
    /// repeat offsets and the content read by index.
    fn reference_dict(bytes: &[u8]) -> Result<DictParts, Error> {
        if bytes.len() < 8 {
            return Err(Error::DictionaryFormat);
        }
        if read_le(bytes, 0, 4)? as u32 != DICT_MAGIC {
            return Ok((0, bytes.to_vec(), HufTable::boxed(), SeqState::boxed()));
        }
        let id = read_le(bytes, 4, 4)? as u32;
        if id == 0 {
            return Err(Error::DictionaryFormat);
        }
        let (mut huf, mut seq) = (HufTable::boxed(), SeqState::boxed());
        let mut p = 8;
        p += huf::read_tree(&bytes[p..], &mut huf).map_err(|_| Error::DictionaryFormat)?;
        let rest = bytes.get(p..).ok_or(Error::DictionaryFormat)?;
        p += seq.read_dict_tables(rest).map_err(|_| Error::DictionaryFormat)?;
        let content = bytes.len().checked_sub(p + 12).ok_or(Error::DictionaryFormat)?;
        for (i, rep) in seq.reps.iter_mut().enumerate() {
            let r = read_le(bytes, p + 4 * i, 4)? as u32;
            if r == 0 || r as usize > content {
                return Err(Error::DictionaryFormat);
            }
            *rep = r;
        }
        Ok((id, bytes[p + 12..].to_vec(), huf, seq))
    }

    /// `Dictionary::new` against the reader it replaced on both test dictionaries (formatted and raw),
    /// cut at every length and with each of their first 400 bytes changed: the same error, or the same id,
    /// content, repeat offsets and tables.
    #[test]
    fn dictionary_agrees_with_reference() {
        let check = |bytes: &[u8]| match (Dictionary::new(bytes), reference_dict(bytes)) {
            (Ok(d), Ok((id, content, huf, seq))) => {
                assert_eq!((d.id, &d.content), (id, &content));
                match (&d.huf, &d.seq) {
                    (Some(h), Some(s)) => assert!(h.dt == huf.dt && s.same_tables(&seq)),
                    (None, None) => assert_eq!(id, 0),
                    _ => panic!("one table without the other"),
                }
            }
            (Err(a), Err(b)) => assert_eq!(a, b),
            (a, b) => panic!("new ok {}, reference ok {}, {} bytes", a.is_ok(), b.is_ok(), bytes.len()),
        };
        for name in ["samples.dict", "base.txt"] {
            let d = embed::file(name);
            for n in 0..=d.len() {
                check(&d[..n]);
            }
            let mut m = d.to_vec();
            for i in 0..d.len().min(400) {
                for x in [1u8, 0x10, 0x80, 0xFF] {
                    m[i] ^= x;
                    check(&m);
                    m[i] ^= x;
                }
            }
        }
    }

    /// The one-shot functions and a reused decoder's methods, into a new vector and onto an existing one,
    /// agree on every corpus frame, on prefixes of it, and on it after a skippable frame and twice over:
    /// the same output or the same error, with the existing vector's bytes kept and nothing added on error.
    #[test]
    fn entry_points_agree() {
        let mut dec = Decoder::new();
        let mut n = 0;
        for (p, data, dict) in frames() {
            let mut inputs = Vec::new();
            for cut in (0..data.len()).step_by(data.len() / 61 + 1).chain([data.len() - 1, data.len()]) {
                inputs.push(data[..cut].to_vec());
            }
            let mut skip = SKIP_MAGIC.to_le_bytes().to_vec();
            skip.extend_from_slice(&3u32.to_le_bytes());
            skip.extend_from_slice(&[1, 2, 3]);
            inputs.push([&skip[..], &data[..]].concat());
            inputs.push([&data[..], &data[..]].concat());
            for input in &inputs {
                let want = match &dict {
                    Some(d) => decompress_with_dict(input, d),
                    None => decompress(input),
                };
                let into = |f: &mut dyn FnMut(&mut Vec<u8>) -> Result<usize, Error>| {
                    let mut out = b"keep".to_vec();
                    let r = f(&mut out);
                    assert_eq!(&out[..4], b"keep", "{p}");
                    r.map(|k| {
                        assert_eq!(k, out.len() - 4, "{p}");
                        out[4..].to_vec()
                    })
                };
                let got = match &dict {
                    Some(d) => [
                        dec.decompress_with_dict(input, d),
                        into(&mut |o| decompress_into_with_dict(input, d, o)),
                        into(&mut |o| dec.decompress_into_with_dict(input, d, o)),
                    ],
                    None => [dec.decompress(input), into(&mut |o| decompress_into(input, o)), into(&mut |o| dec.decompress_into(input, o))],
                };
                for g in got {
                    assert_eq!(g, want, "{p} at {} bytes", input.len());
                }
                n += 1;
            }
        }
        assert!(n > 100);
    }

    #[test]
    fn generic_and_bmi2_paths_agree() {
        let mut n = 0;
        for (p, data, dict) in frames() {
            let half = Cpu { bmi2: Cpu::detect().bmi2, avx2: false };
            let mut outs = [Vec::new(), Vec::new(), Vec::new()];
            let mut rs = [Ok(0), Ok(0), Ok(0)];
            for (i, cpu) in [Cpu::PLAIN, Cpu::detect(), half].into_iter().enumerate() {
                let mut d = Decoder::on(cpu);
                rs[i] = match &dict {
                    Some(k) => d.decompress_into_with_dict(&data, k, &mut outs[i]),
                    None => d.decompress_into(&data, &mut outs[i]),
                };
            }
            assert_eq!(rs[0], rs[1], "{p}");
            assert_eq!(rs[0], rs[2], "{p}");
            assert!(outs[0] == outs[1] && outs[0] == outs[2], "{p}");
            n += rs[0].is_ok() as usize;
        }
        assert!(n >= 17);
    }

    /// `header` against a byte-by-byte reading of the format, for every descriptor, every input length up
    /// to the longest header and two start offsets.
    #[test]
    fn header_fields() {
        fn reference(input: &[u8], mut p: usize) -> Result<(bool, u32, Option<u64>, u64, usize), Error> {
            let fhd = *input.get(p).ok_or(Error::Truncated)?;
            p += 1;
            if fhd & 0x08 != 0 {
                return Err(Error::ReservedBit);
            }
            let single = fhd & 0x20 != 0;
            let did_size = [0, 1, 2, 4][(fhd & 3) as usize];
            let fcs_size = [single as usize, 2, 4, 8][(fhd >> 6) as usize];
            let mut window = 0;
            if !single {
                let wd = *input.get(p).ok_or(Error::Truncated)?;
                p += 1;
                let base = 1u64 << (10 + (wd >> 3) as u32);
                window = base + (base >> 3) * (wd & 7) as u64;
            }
            let did = read_le(input, p, did_size)? as u32;
            p += did_size;
            let fcs = if fcs_size == 0 { None } else { Some(read_le(input, p, fcs_size)? + if fcs_size == 2 { 256 } else { 0 }) };
            p += fcs_size;
            if single {
                window = fcs.unwrap_or(0);
            }
            Ok((fhd & 0x04 != 0, did, fcs, window, p))
        }
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for fhd in 0..=255u8 {
            for start in [0usize, 4] {
                let mut bytes = vec![0u8; start + 15];
                bytes[start] = fhd;
                for b in &mut bytes[start + 1..] {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    *b = (seed >> 56) as u8;
                }
                for len in start..=bytes.len() {
                    let input = &bytes[..len];
                    let got = read_header(input, start, &mut MaybeUninit::uninit()).map(|(h, len)| (h.checksum, h.did, h.fcs(), h.window, start + len));
                    assert_eq!(got, reference(input, start), "fhd {fhd:#x} start {start} len {len}");
                }
            }
        }
    }

    /// `content_size` and `dictionary_id` against the index-based readers they replaced (`read_header`,
    /// then the blocks read with `read_le`), on frames and skippable frames cut at every length and with
    /// bytes changed.
    #[test]
    fn content_size_walk() {
        fn reference_did(input: &[u8]) -> Result<u32, Error> {
            let mut p = 0usize;
            loop {
                match read_le(input, p, 4)? as u32 {
                    MAGIC => return Ok(read_header(input, p + 4, &mut MaybeUninit::uninit())?.0.did),
                    m if m & 0xFFFF_FFF0 == SKIP_MAGIC => p = skip_frame(input, p)?,
                    _ => return Err(Error::BadMagic),
                }
            }
        }
        fn reference(input: &[u8]) -> Result<Option<u64>, Error> {
            if input.is_empty() {
                return Err(Error::Truncated);
            }
            let (mut total, mut p) = (0u64, 0usize);
            while p < input.len() {
                match read_le(input, p, 4)? as u32 {
                    MAGIC => {
                        let mut h = MaybeUninit::uninit();
                        let (h, len) = read_header(input, p + 4, &mut h)?;
                        let Some(n) = h.fcs() else {
                            return Ok(None);
                        };
                        total = total.checked_add(n).ok_or(Error::ContentSize)?;
                        p += 4 + len;
                        loop {
                            let bh = read_le(input, p, 3)? as usize;
                            let size = match (bh >> 1) & 3 {
                                1 => 1,
                                3 => return Err(Error::BlockType),
                                _ => bh >> 3,
                            };
                            p += 3 + size;
                            if p > input.len() {
                                return Err(Error::Truncated);
                            }
                            if bh & 1 != 0 {
                                break;
                            }
                        }
                        p += 4 * h.checksum as usize;
                        if p > input.len() {
                            return Err(Error::Truncated);
                        }
                    }
                    m if m & 0xFFFF_FFF0 == SKIP_MAGIC => p = skip_frame(input, p)?,
                    _ => return Err(Error::BadMagic),
                }
            }
            Ok(Some(total))
        }
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let mut inputs: Vec<Vec<u8>> = frames().into_iter().map(|(_, d, _)| d).filter(|d| d.len() <= 4096).collect();
        // Frames of raw, RLE and reserved blocks under every descriptor, with and without a checksum,
        // followed by a skippable frame, and two frames with content sizes that overflow together.
        for fhd in (0..=255u8).filter(|f| f & 0x08 == 0) {
            let mut f = MAGIC.to_le_bytes().to_vec();
            f.push(fhd);
            // SAFETY: a byte.
            let n = unsafe { head_len(&fhd) };
            f.extend((1..n).map(|_| next() as u8));
            for (kind, last) in [(0usize, 0usize), (1, 0), (0, 1)] {
                let size = (next() % 5) as usize;
                let bh = size << 3 | kind << 1 | last;
                f.extend_from_slice(&bh.to_le_bytes()[..3]);
                f.extend((0..if kind == 1 { 1 } else { size }).map(|_| next() as u8));
            }
            f.extend((0..(fhd & 4)).map(|_| next() as u8));
            f.extend_from_slice(&(SKIP_MAGIC | (next() & 15)).to_le_bytes());
            f.extend_from_slice(&3u32.to_le_bytes());
            f.extend_from_slice(&[1, 2, 3]);
            let mut bad = f.clone();
            bad[4 + n] = 6;
            inputs.push(f);
            inputs.push(bad);
        }
        let big = [MAGIC.to_le_bytes().as_slice(), &[0xE0], &u64::MAX.to_le_bytes(), &[1, 0, 0]].concat();
        inputs.push([big.as_slice(), &big].concat());
        let mut n = 0;
        for input in &inputs {
            for len in 0..=input.len() {
                let mut cut = input[..len].to_vec();
                assert_eq!(content_size(&cut), reference(&cut), "{cut:x?}");
                assert_eq!(dictionary_id(&cut), reference_did(&cut), "{cut:x?}");
                if len > 0 {
                    let at = next() as usize % len;
                    cut[at] ^= 1 << (next() % 8);
                    assert_eq!(content_size(&cut), reference(&cut), "{cut:x?}");
                    assert_eq!(dictionary_id(&cut), reference_did(&cut), "{cut:x?}");
                }
                n += 1;
            }
        }
        assert!(n > 10_000, "{n}");
    }

    #[test]
    fn checksum_verified() {
        let blocks = [((3 << 3) | 1) as u8, 0, 0, b'a', b'b', b'c'];
        let mut f = MAGIC.to_le_bytes().to_vec();
        f.extend_from_slice(&[0x24, 3]);
        f.extend_from_slice(&blocks);
        f.extend_from_slice(&(xxh64::xxh64(b"abc", 0) as u32).to_le_bytes());
        assert_eq!(decompress(&f).unwrap(), b"abc");
        let n = f.len();
        f[n - 1] ^= 1;
        assert_eq!(decompress(&f), Err(Error::Checksum));
    }
}
