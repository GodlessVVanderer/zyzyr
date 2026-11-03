use alloc::boxed::Box;
use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use core::{ptr, slice};

#[cfg(target_has_atomic = "64")]
use core::sync::atomic::AtomicU64;

#[cfg(not(target_has_atomic = "64"))]
mod atomic_u64_compat {
    use core::cell::UnsafeCell;

    #[repr(transparent)]
    pub struct AtomicU64(UnsafeCell<u64>);

    unsafe impl Sync for AtomicU64 {}

    impl AtomicU64 {
        #[inline(always)]
        pub fn load(&self, _: core::sync::atomic::Ordering) -> u64 {
            unsafe { core::ptr::read_volatile(self.0.get()) }
        }

        #[inline(always)]
        pub fn store(&self, v: u64, _: core::sync::atomic::Ordering) {
            unsafe { core::ptr::write_volatile(self.0.get(), v) }
        }
    }
}

#[cfg(not(target_has_atomic = "64"))]
use atomic_u64_compat::AtomicU64;

use crate::bits::Back;
use crate::cpu::multiversion;
use crate::fse::{build_with, read_ncount, Scratch};
use crate::Error;

/// The two-symbol table comes first and the FSE build block first in `Tree`, so the table, the tree
/// work block and the build block share one address: `read_tree` holds one pointer across its calls.
#[repr(C)]
pub(crate) struct HufTable {
    /// The two-symbol table; while `read_tree` runs, its first bytes hold the `Tree` work block
    /// (`read_tree` clears `pair` first, so `pairs` rewrites all of it before the next two-symbol decode).
    dt2: Pairs,
    pub dt: [u16; 2048],
    w: Work,
    pub pair: bool,
}

/// The two-symbol table, aligned for the `Tree` work block that shares its memory (its `Rank` is
/// line-aligned). Measured on its own: this alignment made w10 0.57% faster (20 pinned rounds).
#[repr(C, align(64))]
struct Pairs([u32; 2048]);

const _: () = assert!(size_of::<Tree>() <= size_of::<Pairs>() && align_of::<Tree>() <= align_of::<Pairs>());

/// Work block of `read_tree`: the weights, the FSE table that codes them and its build arrays (with
/// the normalized counts), the rank record of the new tree, and the job: where the weight description
/// lies and what was read of it. Every field is written before it is read in each call, except `w`,
/// which is cleared once per tree. `read_tree` holds nothing but the table across its calls: each
/// phase reads its inputs from here.
#[repr(C)]
struct Tree {
    s: Scratch,
    rank: Rank,
    w: [u8; 256],
    fse: [Cell; 64],
    /// A weight stream shorter than 8 bytes, zero above its last byte.
    pad: u64,
    /// Address and length of the FSE weight description (after the header byte), and the bytes of
    /// its NCount header: the weight stream is `data + head .. data + size`.
    data: usize,
    size: usize,
    head: usize,
    /// Weight count (the last weight is implied) and bytes of the whole tree description.
    n: usize,
    used: u32,
    /// Why `tree` failed: `E_HUF`, `E_FSE` or `E_BITS`.
    err: u32,
    /// First byte of the weight stream as `weights` reads it (the stream or `pad`), read by its refill.
    from: usize,
}

/// A cell of the weight table: the next state's base, the symbol, the bit count and 63 minus the bit
/// count, each loaded where `weights` uses it, so no arithmetic stands between the load and the two
/// shifts of a read. A state is below the table size (at most 64), so the base fits a byte.
#[derive(Clone, Copy)]
#[repr(C)]
struct Cell {
    base: u8,
    sym: u8,
    nb: u8,
    inv: u8,
}

impl Tree {
    /// The weight table from the counts in `s` (false: `Error::Fse`).
    #[inline(always)]
    fn table(&mut self) -> bool {
        let Tree { s, fse, .. } = self;
        build_with(s, fse.len(), |i, sym, nb, base| fse[i] = Cell { base: base as u8, sym, nb, inv: 63u8.wrapping_sub(nb) }).is_ok()
    }

    #[inline(always)]
    fn fail(&mut self, e: u32) -> bool {
        self.err = e;
        false
    }
}

impl HufTable {
    /// Built in place: the allocation comes zeroed, which is the whole empty table (built as a value,
    /// its 12 KB went through the stack into the box).
    pub(crate) fn boxed() -> Box<HufTable> {
        // SAFETY: every field is integers, arrays of integers, atomic integers or a `bool`, for which
        // zero bytes are valid; zero is the empty table and the cleared `pair` flag.
        unsafe { Box::<HufTable>::new_zeroed().assume_init() }
    }

    /// The tree work block and the single-symbol table.
    fn parts(&mut self) -> (&mut Tree, &mut [u16; 2048]) {
        // SAFETY: `Pairs` is large and aligned enough for `Tree` (asserted above) and every bit
        // pattern is a valid `Tree` (integers and arrays of integers only); `dt2` and `dt` are
        // disjoint fields of `self`.
        (unsafe { &mut *self.dt2.0.as_mut_ptr().cast::<Tree>() }, &mut self.dt)
    }

    fn tree(&mut self) -> &mut Tree {
        self.parts().0
    }

    /// `floor` is the start of the buffer `src` lies in; the 4-stream decoder reads from `floor` up.
    pub(crate) fn bind(&self, floor: *const u8, src: &[u8], dst: &mut [u8]) {
        for (r, v) in self.w.job.iter().zip([src.as_ptr() as usize, src.len(), dst.as_mut_ptr() as usize, dst.len()]) {
            r.store(v, Relaxed);
        }
        self.w.floor.store(floor as usize, Relaxed);
    }

    fn pairs(&mut self) {
        for (i, d) in self.dt2.0.iter_mut().enumerate() {
            let e0 = self.dt[i] as u32;
            let l0 = e0 & 0xFF;
            let e1 = self.dt[(i << l0) & 2047] as u32;
            let l1 = e1 & 0xFF;
            *d = if l0 + l1 <= 11 { (l0 + l1) | 2 << 8 | (e0 >> 8) << 16 | (e1 >> 8) << 24 } else { l0 | 1 << 8 | (e0 >> 8) << 16 };
        }
        self.pair = true;
    }
}

const PAIR_MIN: usize = 12 * 1024;

/// Error codes `tree` leaves in the block.
const E_HUF: u32 = 0;
const E_FSE: u32 = 1;
const E_BITS: u32 = 2;

/// Reads a tree description and builds its tables; returns the bytes it takes.
#[cfg(test)]
pub(crate) fn read_tree(src: &[u8], t: &mut HufTable) -> Result<usize, Error> {
    if tree(src, t) {
        return Ok(t.tree_used());
    }
    Err(t.tree_error())
}

impl HufTable {
    /// Bytes of the tree description the last `tree` read.
    #[inline(always)]
    pub(crate) fn tree_used(&mut self) -> usize {
        self.tree().used as usize
    }

    /// Why the last `tree` failed.
    #[inline(always)]
    pub(crate) fn tree_error(&mut self) -> Error {
        match self.tree().err {
            E_FSE => Error::Fse,
            E_BITS => Error::Bitstream,
            _ => Error::Huffman,
        }
    }
}

/// `read_tree` with a flag result: the byte count and the error code stay in the block. The only value
/// held across the calls is the table (the tree and build blocks share its address): every phase takes
/// its inputs from the block, every value needed after a call is stored there before it, and every
/// failure stores its code as an immediate (a `Result` here kept its error tag in a callee-saved
/// register across all calls).
#[inline(never)]
pub(crate) fn tree(src: &[u8], t: &mut HufTable) -> bool {
    t.pair = false;
    let k = t.tree();
    let Some(&h) = src.first() else { return k.fail(E_HUF) };
    let h = h as usize;
    k.w = [0; 256];
    let n = if h >= 128 {
        let n = h - 127;
        let bytes = n.div_ceil(2);
        let Some(data) = src.get(1..1 + bytes) else { return k.fail(E_HUF) };
        for (i, x) in k.w[..n].iter_mut().enumerate() {
            let b = data[i >> 1];
            *x = if i & 1 == 0 { b >> 4 } else { b & 15 };
        }
        k.used = 1 + bytes as u32;
        n
    } else {
        let Some(data) = src.get(1..1 + h).filter(|_| h != 0) else { return k.fail(E_HUF) };
        (k.data, k.size, k.used) = (data.as_ptr() as usize, h, 1 + h as u32);
        // A malformed NCount header and a table log above 6 both fail the build (Error::Fse).
        read_ncount(data, 15, &mut k.s);
        k.head = k.s.used as usize;
        if !k.table() {
            return k.fail(E_FSE);
        }
        if k.head > k.size {
            return k.fail(E_HUF);
        }
        match weights(k) {
            0 => return k.fail(E_BITS),
            1 => return k.fail(E_HUF),
            n => n as usize,
        }
    };
    build(t, n) || t.tree().fail(E_HUF)
}

/// The FSE-coded weights `k.data + k.head .. k.data + k.size` at table log `k.s.log`: two states over
/// one backward stream. `cur` counts the stream's bits below the next read and `c` holds the bits below
/// it, left-aligned: a read takes the top bits of `c` and shifts them out, and `cur` turns negative
/// exactly when a read passes the stream start (the `used > 64` of a reader whose window stops at the
/// first byte). The refill puts a window under `cur` from the cursor alone, with at least 57 bits
/// below it (or the stream start), so it runs once per two pairs (24 bits at most); the block holds the
/// window's base, a memory operand. A read shifts by the cell's bit count and by 63 minus it, both
/// loaded from the cell. Returns the weight count (at least 2), or 0 for a stream whose last
/// byte is zero (`Error::Bitstream`) and 1 for every other error (`Error::Huffman`): one word in a
/// register on every target.
#[inline(never)]
fn weights(k: &mut Tree) -> u32 {
    let (len, log) = (k.size - k.head, k.s.log);
    // SAFETY: `read_tree` (or a test) set `data .. data + size` inside a live slice, `head <= size`.
    let stream = unsafe { slice::from_raw_parts((k.data + k.head) as *const u8, len) };
    let Tree { w, fse, pad, from, .. } = k;
    if len == 0 || stream[len - 1] == 0 {
        return 0;
    }
    let s = if len >= 8 {
        stream.as_ptr()
    } else {
        *pad = stream.iter().rev().fold(0u64, |v, &b| v << 8 | b as u64).to_le();
        (pad as *const u64).cast::<u8>()
    };
    *from = s as usize;
    // The last 8 bytes; its top set bit is the end marker, the bits below it the first read.
    let top = len.max(8) - 8;
    let bits = unsafe { load(s.add(top)) };
    let lz = bits.leading_zeros();
    let mut cur = (8 * top) as isize + 63 - lz as isize;
    let mut c = bits.wrapping_shl(lz + 1);
    // `n` bits, `inv` = 63 - n. `keep` shifts the read bits out of `c`; `last` is the read before a
    // refill, which rebuilds `c`.
    macro_rules! read {
        (keep $n:expr, $inv:expr) => {{
            let n: u32 = $n;
            let v = read!(last n, $inv);
            c = c.wrapping_shl(n);
            v
        }};
        (last $n:expr, $inv:expr) => {{
            cur -= $n as isize;
            (c >> 1).wrapping_shr($inv) as usize
        }};
    }
    macro_rules! refill {
        () => {{
            let pos = (cur - 57).max(0) as usize >> 3;
            // SAFETY: `pos + 8 <= len.max(8)`: `cur <= 8 * top + 63`; `from` is `s`, written above.
            c = unsafe { load((ptr::read(from) + pos) as *const u8) }.wrapping_shl((8 * pos as isize - cur) as u32);
        }};
    }
    // The last byte is not zero, so at most 8 bits are taken: two reads fit before the first refill.
    let mut s1 = read!(keep log, 63 - log);
    let mut s2 = read!(last log, 63 - log);
    if cur < 0 {
        return 1;
    }
    let mut n = 0usize;
    macro_rules! field {
        ($e:ident . sym) => {
            unsafe { ptr::read(&raw const (*$e).sym) }
        };
        ($e:ident . base) => {
            (unsafe { ptr::read(&raw const (*$e).base) } as usize)
        };
        ($e:ident . $f:ident) => {
            (unsafe { ptr::read(&raw const (*$e).$f) } as u32)
        };
    }
    macro_rules! pair {
        ($($refill:ident)?; $tail:ident) => {{
            if n >= 254 {
                return 1;
            }
            $($refill!();)?
            // SAFETY: a state is below the table size (at most 64): a first read takes `log` bits, and a
            // cell's `base + 2^nb` is at most the size (its state `x` in `[c, 2c)` for a count `c`,
            // `base = x << nb - size`, `nb = log - hb(x)`); every cell below the size was written by the
            // build `tree` checked.
            // Each field is read where used, so no cell waits in registers.
            let (e1, e2) = unsafe { (fse.as_ptr().add(s1), fse.as_ptr().add(s2)) };
            w[n & 255] = field!(e1.sym);
            s1 = read!(keep field!(e1.nb), field!(e1.inv)) + field!(e1.base);
            if cur < 0 {
                w[(n + 1) & 255] = field!(e2.sym);
                n += 2;
                break;
            }
            w[(n + 1) & 255] = field!(e2.sym);
            s2 = read!($tail field!(e2.nb), field!(e2.inv)) + field!(e2.base);
            n += 2;
            if cur < 0 {
                // SAFETY: as above.
                w[n & 255] = unsafe { fse.get_unchecked(s1) }.sym;
                n += 1;
                break;
            }
        }};
    }
    loop {
        pair!(refill; keep);
        pair!(; keep);
        pair!(; keep);
        pair!(; last);
    }
    n as u32
}

/// Symbols per quarter of `w[..=n]`.
#[inline(always)]
fn quarter(n: usize) -> usize {
    16 << (n >> 6).min(2)
}

/// The tables from the `n` weights in the tree block (false: `Error::Huffman`). The weight count is
/// stored in the block and read back by `order`, the bit count goes to `fill` in the rank record,
/// so nothing is held across the leaf calls but the table.
#[inline(always)]
fn build(t: &mut HufTable, n: usize) -> bool {
    if n > 255 {
        return false;
    }
    let k = t.tree();
    k.n = n;
    // Four quarters of q symbols cover w[..=n]; w is zero from n on (read_tree zeroes it).
    let ok = match quarter(n) {
        16 => tally::<16>(k),
        32 => tally::<32>(k),
        _ => tally::<64>(k),
    };
    if !ok || !order(t.tree()) {
        return false;
    }
    let k = t.tree();
    match quarter(k.n) {
        16 => rank::<16>(k),
        32 => rank::<32>(k),
        _ => rank::<64>(k),
    }
    unsafe { fill(t) };
    true
}

/// From the weight counts of `tally`: the implied last weight, the counts per weight, each quarter's
/// first table-order index per weight and the bit count; false if the weights make no complete code.
/// A leaf, so its temporaries (and the weight array's address) are not held across `read_tree`'s calls.
#[inline(never)]
fn order(k: &mut Tree) -> bool {
    let Tree { rank: r, w, n, .. } = k;
    let total: u32 = (1..12).map(|x| r.part.iter().map(|p| p[x] as u32).sum::<u32>() << (x - 1)).sum();
    if total == 0 {
        return false;
    }
    let max_bits = 32 - total.leading_zeros();
    if max_bits > 11 {
        return false;
    }
    let rest = (1u32 << max_bits) - total;
    if !rest.is_power_of_two() {
        return false;
    }
    let last = rest.trailing_zeros() + 1;
    // The weight count is read here, after the sum: read first, it and the quarter took two
    // registers across the sum's eleven terms (8 values against 7 volatile registers).
    let (n, q) = ((*n).min(255), quarter(*n));
    w[n] = last as u8;
    let p = &mut r.part[(n / q) & 3];
    p[0] -= 1;
    // rest < 2^11, so last <= 11; the mask only drops the bound check (its panic call made the leaf
    // keep a frame on aarch64).
    p[last as usize & 15] += 1;
    // Counts per weight, and each quarter's first table-order index per weight.
    let mut at = 0u16;
    for x in 0..16 {
        let c0 = at;
        for p in r.part.iter_mut() {
            (p[x], at) = (at, at + p[x]);
        }
        r.cnt[x] = at - c0;
    }
    let scale = 11 - max_bits as usize;
    // max_bits <= 11, so the indices stay inside `cnt` (a loop, not a `1..=max_bits` slice: that
    // range indexes through a call).
    let acc: usize = (1..max_bits as usize + 1).map(|x| (r.cnt[x & 15] as usize) << (scale + x - 1)).sum();
    r.nb = max_bits as u16 + 1;
    acc == 1 << 11
}

/// Table entries of the symbols in table order (weight 0 first, then ascending weight, then ascending
/// symbol) and the symbol count per weight: the table regions of the weights follow each other.
/// `part` holds per quarter of the symbols the count per weight, then the next index per weight: four
/// counters per weight, so symbols of one weight in a row do not wait on each other's count.
/// Aligned to a line, so its rows lie in the same lines whatever precedes it in `Tree` (behind the
/// FSE build block, unaligned, its `part` rows crossed lines).
#[repr(C, align(64))]
struct Rank {
    ent: [u16; 256],
    cnt: [u16; 16],
    part: [[u16; 16]; 4],
    /// max_bits + 1: the entry's bit count is nb - weight.
    nb: u16,
}

/// Weight counts per quarter of `w[..4 * Q]`; false if a weight is above 11.
#[inline(never)]
fn tally<const Q: usize>(k: &mut Tree) -> bool {
    let Tree { rank: r, w, .. } = k;
    r.part = [[0; 16]; 4];
    let mut high = 0u32;
    for i in 0..Q {
        for (k, p) in r.part.iter_mut().enumerate() {
            let x = w[(k * Q + i) & 255];
            high |= x as u32 + 4;
            p[x as usize & 15] += 1;
        }
    }
    high < 16
}

/// Places the table entry of every symbol of `w[..4 * Q]` at its table-order index. `r.nb` is read
/// where used (a memory operand), so the loop holds no register for it.
#[inline(never)]
fn rank<const Q: usize>(k: &mut Tree) {
    let Tree { rank: r, w, .. } = k;
    for i in 0..Q {
        for k in 0..4 {
            let s = k * Q + i;
            let x = w[s & 255] as usize & 15;
            let at = r.part[k][x];
            r.part[k][x] = at + 1;
            let nb = unsafe { ptr::read(&r.nb) };
            r.ent[at as usize & 255] = ((s as u16) << 8).wrapping_add(nb).wrapping_sub(x as u16);
        }
    }
}

/// Writes the 2048 entries region by region from the rank record (max_bits is `nb - 1`). Safety: the
/// record comes from `build`, which checked that the regions add up to 2048 entries. The heaviest
/// weight present has regions of at least 8 entries (at most 256 symbols share 2^max_bits), so the
/// 8-entry store of a short region only runs into the next region, which is written after it.
#[inline(never)]
unsafe fn fill(t: &mut HufTable) {
    let (k, dt) = t.parts();
    let r = &k.rank;
    let max_bits = (r.nb as u32).wrapping_sub(1) & 15;
    let mut q = dt.as_mut_ptr().cast::<u8>();
    let mut p = r.ent.as_ptr().wrapping_add(r.cnt[0] as usize);
    let mut c = r.cnt.as_ptr();
    // Region size in bytes of weight 1, doubling per weight: 2048 at weight max_bits, then the end.
    let mut size = 2usize << (11 - max_bits);
    while size != 4096 {
        unsafe {
            c = c.add(1);
            let mut k = *c as usize;
            if size < 16 {
                while k != 0 {
                    v8::put(q, v8::splat(*p));
                    q = q.add(size);
                    p = p.add(1);
                    k -= 1;
                }
            } else {
                while k != 0 {
                    k -= 1;
                    let v = v8::splat(*p);
                    let stop = q.add(size);
                    loop {
                        v8::put(q, v);
                        q = q.add(16);
                        if q == stop {
                            break;
                        }
                    }
                    p = p.add(1);
                }
            }
        }
        size <<= 1;
    }
}

/// Eight copies of a table entry in one 16-byte vector store.
#[cfg(target_arch = "x86_64")]
mod v8 {
    use core::arch::x86_64::{__m128i, _mm_set1_epi16, _mm_storeu_si128};

    #[inline(always)]
    pub(super) fn splat(e: u16) -> __m128i {
        // SSE2 is part of every x86_64 target.
        unsafe { _mm_set1_epi16(e as i16) }
    }

    #[inline(always)]
    pub(super) unsafe fn put(q: *mut u8, v: __m128i) {
        unsafe { _mm_storeu_si128(q.cast(), v) }
    }
}

#[cfg(target_arch = "aarch64")]
mod v8 {
    use core::arch::aarch64::{uint16x8_t, vdupq_n_u16, vst1q_u16};

    #[inline(always)]
    pub(super) fn splat(e: u16) -> uint16x8_t {
        // NEON is part of every aarch64 target.
        unsafe { vdupq_n_u16(e) }
    }

    #[inline(always)]
    pub(super) unsafe fn put(q: *mut u8, v: uint16x8_t) {
        unsafe { vst1q_u16(q.cast(), v) }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod v8 {
    use core::arch::wasm32::{u16x8_splat, v128, v128_store};

    #[inline(always)]
    pub(super) fn splat(e: u16) -> v128 {
        u16x8_splat(e)
    }

    #[inline(always)]
    pub(super) unsafe fn put(q: *mut u8, v: v128) {
        unsafe { v128_store(q.cast(), v) }
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
mod v8 {
    use core::ptr;

    #[inline(always)]
    pub(super) fn splat(e: u16) -> u64 {
        e as u64 * 0x0001_0001_0001_0001
    }

    #[inline(always)]
    pub(super) unsafe fn put(q: *mut u8, v: u64) {
        unsafe {
            ptr::write_unaligned(q.cast::<u64>(), v);
            ptr::write_unaligned(q.add(8).cast::<u64>(), v);
        }
    }
}

const SHIFT: u32 = 53;

const GROUP: usize = 5;

const GROUP_BYTES: usize = 7;

/// Takes the table, not its single-symbol array: the callers hold the table pointer, and `dt` lies
/// past the two-symbol table, so passing `&t.dt` held a second pointer.
fn tail(t: &HufTable, r: &mut Back, out: &mut [u8]) -> Result<(), Error> {
    let dt = &t.dt;
    for o in out.iter_mut() {
        r.refill();
        let e = dt[(r.bits.wrapping_shl(r.used) >> SHIFT) as usize];
        r.used += (e & 0xFF) as u32;
        *o = (e >> 8) as u8;
    }
    if r.finish() {
        Ok(())
    } else {
        Err(Error::Huffman)
    }
}

#[inline(always)]
unsafe fn load(p: *const u8) -> u64 {
    unsafe { u64::from_le(ptr::read_unaligned(p.cast::<u64>())) }
}

/// Work block of the stream decoders, one array per field and lane index `j` in 0..4. The read
/// positions and the loop limits, touched on every refill, are its first 64 bytes. Measured: aligning
/// the block to 64 bytes made w10 decoding slower (6.991 vs 6.941 cycles/B pinned).
#[repr(C)]
struct Work {
    p: [AtomicUsize; 4],
    lim: [AtomicUsize; 4],
    c: [AtomicU64; 4],
    o: [AtomicUsize; 4],
    e: [AtomicUsize; 4],
    start: [AtomicUsize; 4],
    len: [AtomicUsize; 4],
    job: [AtomicUsize; 4],
    floor: AtomicUsize,
}

impl Work {
    #[inline(always)]
    fn open(&self, j: usize, start: usize, len: usize, o: usize, e: usize) {
        for (r, v) in [&self.start[j], &self.len[j], &self.o[j], &self.e[j]].into_iter().zip([start, len, o, e]) {
            r.store(v, Relaxed);
        }
    }

    #[inline(always)]
    unsafe fn enter(&self, j: usize) {
        let p = self.start[j].load(Relaxed) + self.len[j].load(Relaxed) - 8;
        let bits = unsafe { load(p as *const u8) };
        self.p[j].store(p, Relaxed);
        self.c[j].store((bits | 1) << (bits.leading_zeros() + 1), Relaxed);
    }

    #[inline(always)]
    unsafe fn src<'a>(&self, j: usize) -> &'a [u8] {
        unsafe { slice::from_raw_parts(self.start[j].load(Relaxed) as *const u8, self.len[j].load(Relaxed)) }
    }

    #[inline(always)]
    unsafe fn out<'a>(&self, j: usize) -> &'a mut [u8] {
        let o = self.o[j].load(Relaxed);
        unsafe { slice::from_raw_parts_mut(o as *mut u8, self.e[j].load(Relaxed) - o) }
    }
}

/// Lane `j` alone from where it stands to the end of its segment (false: `Error::Huffman`). The loop
/// reads the block floor and the segment end where it tests them (memory operands), and the lane index
/// waits in the block, so the loop holds the table, the read position, the container and the output in
/// volatile registers (the symbol store from a high byte register needs a base register without a REX
/// prefix: with the index held as well, the output took a callee-saved one). The floor is the start of
/// the block, so on a valid stream the loop stops at most 4 symbols before the end, and `close` decodes
/// those with a refill that never reads below the floor and checks the end.
#[inline(never)]
unsafe fn finish(t: &HufTable, j: usize) -> bool {
    let (w, dp, j) = (&t.w, t.dt.as_ptr(), j & 3);
    let (mut p, mut c, mut o) = (w.p[j].load(Relaxed), w.c[j].load(Relaxed), w.o[j].load(Relaxed) as *mut u8);
    // The last output position a whole group starts from: the test compares the output pointer itself
    // (a test on `e - o` or `o + 5` became a second induction variable, one register more).
    w.lim[1].store(w.e[j].load(Relaxed).saturating_sub(GROUP), Relaxed);
    w.lim[2].store(j, Relaxed);
    while p >= w.floor.load(Relaxed) + GROUP_BYTES && o as usize <= w.lim[1].load(Relaxed) {
        unsafe {
            for k in 0..GROUP {
                *o.add(k) = pick(&mut c, dp);
            }
            o = o.add(GROUP);
            let s = c.trailing_zeros();
            p -= (s >> 3) as usize;
            c = (load(p as *const u8) | 1) << (s & 7);
        }
    }
    let j = w.lim[2].load(Relaxed) & 3;
    w.p[j].store(p, Relaxed);
    w.c[j].store(c, Relaxed);
    w.o[j].store(o as usize, Relaxed);
    unsafe { close(t, j) }
}

#[cold]
#[inline(never)]
unsafe fn short(t: &HufTable, lanes: usize) -> Result<(), Error> {
    for j in 0..lanes.min(4) {
        let mut pad = [0u8; 8];
        let mut b = Back::new(unsafe { t.w.src(j) }, &mut pad)?;
        tail(t, &mut b, unsafe { t.w.out(j) })?;
    }
    Ok(())
}

multiversion! {
    pub(crate) unsafe fn decode1(cpu: Cpu, t: &HufTable) -> Result<(), Error> = decode1_impl;
}

multiversion! {
    pub(crate) unsafe fn decode4(cpu: Cpu, t: &mut HufTable) -> Result<(), Error> = decode4_impl;
}

#[inline(always)]
unsafe fn pick(c: &mut u64, dt: *const u16) -> u8 {
    let e = unsafe { *dt.add((*c >> SHIFT) as usize) };
    *c = c.wrapping_shl(e as u32);
    (e >> 8) as u8
}

#[inline(always)]
unsafe fn pick2(c: &mut u64, o: &mut *mut u8, dt2: *const u32) {
    unsafe {
        let e = *dt2.add((*c >> SHIFT) as usize);
        ptr::write_unaligned(o.cast::<u16>(), ((e >> 16) as u16).to_le());
        *c = c.wrapping_shl(e);
        *o = o.add(((e >> 8) & 0xFF) as usize);
    }
}

#[inline(always)]
fn reload(slot: &AtomicUsize, c: u64) -> (u64, usize) {
    let t = c.trailing_zeros();
    let p = slot.load(Relaxed) - (t >> 3) as usize;
    slot.store(p, Relaxed);
    ((unsafe { load(p as *const u8) } | 1) << (t & 7), p)
}

/// Refill that never reads below `floor`: once the step would cross it, the window stays at `floor`
/// and the consumed count grows. A container whose marker bit would leave the word reads 0 from then on.
#[inline(always)]
fn clamp(slot: &AtomicUsize, floor: usize, c: u64) -> u64 {
    let t = c.trailing_zeros() as usize;
    let p = slot.load(Relaxed);
    let step = (t >> 3).min(p - floor);
    let p = p - step;
    slot.store(p, Relaxed);
    let s = t - 8 * step;
    let v = unsafe { load(p as *const u8) } | 1;
    if s < 64 { v << s } else { 0 }
}

/// Lane `j` from where the 4-lane loop left it to the end of its segment, then the exact end check of
/// `quad`.
#[inline(never)]
unsafe fn close(t: &HufTable, j: usize) -> bool {
    let (w, dp, j) = (&t.w, t.dt.as_ptr(), j & 3);
    let (mut c, mut o) = (w.c[j].load(Relaxed), w.o[j].load(Relaxed) as *mut u8);
    while o as usize != w.e[j].load(Relaxed) {
        c = clamp(&w.p[j], w.floor.load(Relaxed), c);
        for _ in 0..(w.e[j].load(Relaxed) - o as usize).min(GROUP) {
            unsafe {
                *o = pick(&mut c, dp);
                o = o.add(1);
            }
        }
    }
    ends(&w.p[j], &w.start[j], c)
}

/// Every bit of the lane read and none below its first byte: the window at `p` holds 64 bits, `c` has
/// consumed its top `tz(c)`.
#[inline(always)]
fn ends(p: &AtomicUsize, start: &AtomicUsize, c: u64) -> bool {
    c != 0 && (p.load(Relaxed) + 8).wrapping_mul(8).wrapping_sub(c.trailing_zeros() as usize) == start.load(Relaxed).wrapping_mul(8)
}

#[inline(always)]
unsafe fn dual2(t: &HufTable, ja: usize, jb: usize) {
    let (w, dt2) = (&t.w, t.dt2.0.as_ptr());
    let rec = &w.lim;
    let low = w.job[0].load(Relaxed) + GROUP_BYTES;
    let (ea, eb) = (w.e[ja].load(Relaxed).wrapping_sub(2 * GROUP), w.e[jb].load(Relaxed).wrapping_sub(2 * GROUP));
    let (mut pa, mut pb) = (w.o[ja].load(Relaxed) as *mut u8, w.o[jb].load(Relaxed) as *mut u8);
    if w.p[ja].load(Relaxed) < low || w.p[jb].load(Relaxed) < low || pa as usize > ea || pb as usize > eb {
        return;
    }
    for (r, v) in rec.iter().zip([low, ea, eb]) {
        r.store(v, Relaxed);
    }
    let (mut a, mut b) = (w.c[ja].load(Relaxed), w.c[jb].load(Relaxed));
    unsafe {
        loop {
            for _ in 0..GROUP {
                pick2(&mut a, &mut pa, dt2);
                pick2(&mut b, &mut pb, dt2);
            }
            let (na, qa) = reload(&w.p[ja], a);
            let (nb, qb) = reload(&w.p[jb], b);
            (a, b) = (na, nb);
            if qa.min(qb) < rec[0].load(Relaxed) || pa as usize > rec[1].load(Relaxed) || pb as usize > rec[2].load(Relaxed) {
                break;
            }
        }
    }
    w.c[ja].store(a, Relaxed);
    w.o[ja].store(pa as usize, Relaxed);
    w.c[jb].store(b, Relaxed);
    w.o[jb].store(pb as usize, Relaxed);
}

/// Two-symbol decode of lane `j` while a whole group fits above the block floor and before the
/// segment end, both read where tested, as in `finish`. A leaf of its own: inlined into `finish`, its
/// output chain took three callee-saved registers there.
#[inline(never)]
unsafe fn single2(t: &HufTable, j: usize) {
    let (w, dt2, j) = (&t.w, t.dt2.0.as_ptr(), j & 3);
    let (mut p, mut c, mut o) = (w.p[j].load(Relaxed), w.c[j].load(Relaxed), w.o[j].load(Relaxed) as *mut u8);
    while p >= w.floor.load(Relaxed) + GROUP_BYTES && o as usize + 2 * GROUP <= w.e[j].load(Relaxed) {
        unsafe {
            for _ in 0..GROUP {
                pick2(&mut c, &mut o, dt2);
            }
            let s = c.trailing_zeros();
            p -= (s >> 3) as usize;
            c = (load(p as *const u8) | 1) << (s & 7);
        }
    }
    w.p[j].store(p, Relaxed);
    w.c[j].store(c, Relaxed);
    w.o[j].store(o as usize, Relaxed);
}

#[inline(always)]
unsafe fn decode1_impl(t: &HufTable) -> Result<(), Error> {
    let w = &t.w;
    let (src, len) = (w.job[0].load(Relaxed), w.job[1].load(Relaxed));
    let (o, n) = (w.job[2].load(Relaxed), w.job[3].load(Relaxed));
    if len == 0 || unsafe { *((src + len - 1) as *const u8) } == 0 {
        return Err(Error::Bitstream);
    }
    w.open(0, src, len, o, o + n);
    if len < 8 {
        return unsafe { short(t, 1) };
    }
    unsafe {
        w.enter(0);
        if finish(t, 0) { Ok(()) } else { Err(Error::Huffman) }
    }
}

#[inline(always)]
unsafe fn decode4_impl(t: &mut HufTable) -> Result<(), Error> {
    let n = t.w.job[3].load(Relaxed);
    if n >= PAIR_MIN && !t.pair {
        t.pairs();
    }
    let (t, w) = (&*t, &t.w);
    let (src, len) = (w.job[0].load(Relaxed), w.job[1].load(Relaxed));
    if len < 6 {
        return Err(Error::Huffman);
    }
    let z = |i: usize| unsafe { u16::from_le(ptr::read_unaligned((src + i) as *const u16)) } as usize;
    let mut at = 6;
    let mut long = true;
    let seg = n.div_ceil(4);
    let o = w.job[2].load(Relaxed);
    for j in 0..4 {
        let size = if j < 3 { z(2 * j) } else { len.wrapping_sub(at) };
        if j == 3 && (at > len || 3 * seg > n) {
            return Err(Error::Huffman);
        }
        w.open(j, src + at, size, o + j * seg, if j < 3 { o + (j + 1) * seg } else { o + n });
        at += size;
    }
    for j in 0..4 {
        let (start, size) = (w.start[j].load(Relaxed), w.len[j].load(Relaxed));
        if size == 0 || unsafe { *((start + size - 1) as *const u8) } == 0 {
            return Err(Error::Bitstream);
        }
        long &= size >= 8;
    }
    if !long {
        return unsafe { short(t, 4) };
    }
    for j in 0..4 {
        unsafe { w.enter(j) };
    }
    if n >= PAIR_MIN && t.pair {
        unsafe {
            dual2(t, 0, 1);
            dual2(t, 2, 3);
            rest(t)
        }
    } else {
        let len4 = n - 3 * seg;
        unsafe { quad(t, o, o + len4 - len4 % GROUP, seg) }
    }
}

#[inline(never)]
unsafe fn rest(t: &HufTable) -> Result<(), Error> {
    for j in 0..4 {
        unsafe { single2(t, j) };
        if !unsafe { finish(t, j) } {
            return Err(Error::Huffman);
        }
    }
    Ok(())
}

#[inline(always)]
unsafe fn quad(t: &HufTable, o: usize, group_end: usize, seg: usize) -> Result<(), Error> {
    let (dp, w) = (t.dt.as_ptr(), &t.w);
    let (rec, ln) = (&w.lim, &w.p);
    rec[0].store(w.floor.load(Relaxed), Relaxed);
    rec[1].store(group_end, Relaxed);
    rec[2].store(seg, Relaxed);
    let [mut c0, mut c1, mut c2, mut c3] = [0, 1, 2, 3].map(|j| w.c[j].load(Relaxed));
    let mut q = o as *mut u8;
    // The floor is the start of the block: below lane 0 lie the jump table, the literals header and
    // the tree, below lanes 1-3 the lane before. A valid stream ends every lane at most 7 bytes below
    // its start, so the batch limit stops the loop early only on a corrupt stream or when the literals
    // header and tree take less than 8 bytes; the lanes then end one by one in `close`.
    loop {
        let f = rec[0].load(Relaxed);
        let lanes = (ln[0].load(Relaxed) - f).min(ln[1].load(Relaxed) - f).min((ln[2].load(Relaxed) - f).min(ln[3].load(Relaxed) - f));
        let room = (lanes / GROUP_BYTES).min((rec[1].load(Relaxed) - q as usize) / GROUP);
        if room == 0 {
            break;
        }
        let sg = rec[2].load(Relaxed);
        unsafe {
            for _ in 0..room {
                for k in 0..GROUP {
                    *q.add(k) = pick(&mut c0, dp);
                    *q.add(sg + k) = pick(&mut c1, dp);
                    *q.add(2 * sg + k) = pick(&mut c2, dp);
                    *q.add(3 * sg + k) = pick(&mut c3, dp);
                }
                q = q.add(GROUP);
                c0 = reload(&ln[0], c0).0;
                c1 = reload(&ln[1], c1).0;
                c2 = reload(&ln[2], c2).0;
                c3 = reload(&ln[3], c3).0;
            }
        }
    }
    if q as usize != rec[1].load(Relaxed) {
        let done = q as usize - w.o[0].load(Relaxed);
        for (j, c) in [c0, c1, c2, c3].into_iter().enumerate() {
            w.c[j].store(c, Relaxed);
            w.o[j].store(w.o[j].load(Relaxed) + done, Relaxed);
        }
        return if unsafe { close(t, 0) & close(t, 1) & close(t, 2) & close(t, 3) } { Ok(()) } else { Err(Error::Huffman) };
    }
    // The rest of lane 3 and the up to 3 extra symbols of lanes 0-2, with every bound read back from
    // the block: nothing computed before the loop is held across it.
    rec[3].store(w.e[3].load(Relaxed) - (w.o[3].load(Relaxed) - w.o[0].load(Relaxed)), Relaxed);
    let sg = rec[2].load(Relaxed);
    unsafe {
        while q as usize != rec[3].load(Relaxed) {
            *q = pick(&mut c0, dp);
            *q.add(sg) = pick(&mut c1, dp);
            *q.add(2 * sg) = pick(&mut c2, dp);
            *q.add(3 * sg) = pick(&mut c3, dp);
            q = q.add(1);
        }
        c0 = clamp(&ln[0], rec[0].load(Relaxed), c0);
        c1 = clamp(&ln[1], rec[0].load(Relaxed), c1);
        c2 = clamp(&ln[2], rec[0].load(Relaxed), c2);
        while q as usize != w.e[0].load(Relaxed) {
            *q = pick(&mut c0, dp);
            *q.add(sg) = pick(&mut c1, dp);
            *q.add(2 * sg) = pick(&mut c2, dp);
            q = q.add(1);
        }
    }
    let end = |j: usize, c: u64| ends(&ln[j], &w.start[j], c);
    if end(0, c0) & end(1, c1) & end(2, c2) & end(3, c3) { Ok(()) } else { Err(Error::Huffman) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::Cpu;
    use crate::fse::{build_table, FseEntry};
    use alloc::vec;
    use alloc::vec::Vec;

    /// `build` from the given weights, the rest of the weights zero.
    fn build_from(t: &mut HufTable, weights: &[u8]) -> Result<(), Error> {
        let k = t.tree();
        k.w = [0; 256];
        k.w[..weights.len()].copy_from_slice(weights);
        if build(t, weights.len()) { Ok(()) } else { Err(Error::Huffman) }
    }

    /// `weights` on `stream`, its count and error codes turned back into the weights or the error.
    fn weights_of(k: &mut Tree, stream: &[u8]) -> Result<Vec<u8>, Error> {
        (k.data, k.size, k.head) = (stream.as_ptr() as usize, stream.len(), 0);
        k.w = [0; 256];
        match weights(k) {
            0 => Err(Error::Bitstream),
            1 => Err(Error::Huffman),
            n => Ok(k.w[..n as usize].to_vec()),
        }
    }

    #[test]
    fn spec_weights_to_codes() {
        let mut t = HufTable::boxed();
        build_from(&mut t, &[4, 3, 2, 0, 1]).unwrap();
        assert_eq!(t.parts().0.w[5], 1);
        let code = |s: u16| -> Vec<usize> { (0..16).filter(|&i| t.dt[i << 7] >> 8 == s).collect() };
        assert_eq!(code(4), vec![0]);
        assert_eq!(code(5), vec![1]);
        assert_eq!(code(2), vec![2, 3]);
        assert_eq!(code(1), (4..8).collect::<Vec<_>>());
        assert_eq!(code(0), (8..16).collect::<Vec<_>>());
        assert_eq!(t.dt[0] & 0xFF, 4);
        assert_eq!(t.dt[8 << 7] & 0xFF, 1);
        assert!((0..2048).all(|i| t.dt[i] == t.dt[i & !127]));
    }

    #[test]
    fn spec_stream_with_table_25_codes() {
        let mut t = HufTable::boxed();
        build_from(&mut t, &[4, 3, 2, 0, 1]).unwrap();
        let run = |cpu: Cpu, out: &mut [u8]| {
            let src = [0b0001_0000, 0b0000_1101];
            t.bind(src.as_ptr(), &src, out);
            unsafe { decode1(cpu, &t) }
        };
        let cpu = Cpu::detect();
        let mut out = [0u8; 4];
        run(cpu, &mut out).unwrap();
        assert_eq!(out, [0, 1, 5, 4]);
        let mut out = [0u8; 5];
        assert!(run(cpu, &mut out).is_err());
        let generic = Cpu::PLAIN;
        let mut out = [0u8; 4];
        run(generic, &mut out).unwrap();
        assert_eq!(out, [0, 1, 5, 4]);
    }

    #[test]
    fn direct_weights_header() {
        let src = [128 + 4, 0x43, 0x20, 0x10];
        let mut t = HufTable::boxed();
        assert_eq!(read_tree(&src, &mut t), Ok(4));
        // The implied last weight follows the five read.
        assert_eq!(&t.tree().w[..7], &[4, 3, 2, 0, 1, 1, 0]);
        assert!(read_tree(&src[..3], &mut t).is_err());
    }

    #[test]
    fn rejects_bad_weights() {
        let mut t = HufTable::boxed();
        assert!(build_from(&mut t, &[0, 0, 0]).is_err());
        assert!(build_from(&mut t, &[12]).is_err());
        assert!(build_from(&mut t, &[2, 2, 1]).is_err());
        assert!(read_tree(&[0x10], &mut t).is_err());
        assert!(read_tree(&[], &mut t).is_err());
    }

    /// The weight stream decoded with a refill before every read, the reader `weights` replaced.
    fn weights_per_symbol(fse: &[FseEntry], stream: &[u8], log: u32) -> Result<Vec<u8>, Error> {
        let mut pad = [0u8; 8];
        let mut r = Back::new(stream, &mut pad)?;
        let mask = (1usize << log) - 1;
        let mut s1 = r.read(log) as usize & mask;
        r.refill();
        let mut s2 = r.read(log) as usize & mask;
        if r.used > 64 {
            return Err(Error::Huffman);
        }
        let mut w = Vec::new();
        loop {
            if w.len() >= 254 {
                return Err(Error::Huffman);
            }
            w.push(fse[s1].sym);
            r.refill();
            let e = fse[s1];
            s1 = (e.base as usize + r.read(e.nb as u32) as usize) & mask;
            if r.used > 64 {
                w.push(fse[s2].sym);
                return Ok(w);
            }
            w.push(fse[s2].sym);
            r.refill();
            let e = fse[s2];
            s2 = (e.base as usize + r.read(e.nb as u32) as usize) & mask;
            if r.used > 64 {
                w.push(fse[s1].sym);
                return Ok(w);
            }
        }
    }

    #[test]
    fn weights_match_refill_per_symbol() {
        let mut t = HufTable::boxed();
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let tables = [
            (&[20i16, 12, 8, 6, 5, 4, 3, 2, 1, 1, -1, 1][..], 6),
            (&[10, 8, 6, 4, 2, 1, -1][..], 5),
            (&[-1, -1, 49, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1][..], 6),
        ];
        for (norm, log) in tables {
            let k = t.tree();
            k.s.norm[..norm.len()].copy_from_slice(norm);
            k.s.set(norm.len(), log);
            let mut fse = [FseEntry::default(); 64];
            build_table(&mut k.s, &mut fse).unwrap();
            assert!(k.table());
            for len in 1..160 {
                for _ in 0..40 {
                    let stream: Vec<u8> = (0..len)
                        .map(|_| {
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            x as u8
                        })
                        .collect();
                    let want = weights_per_symbol(&fse, &stream, log);
                    let got = weights_of(k, &stream);
                    assert_eq!(got, want, "log {log} stream {stream:?}");
                    if let Ok(w) = want {
                        assert!(k.w[w.len()..].iter().all(|&b| b == 0));
                    }
                }
            }
        }
    }
}
