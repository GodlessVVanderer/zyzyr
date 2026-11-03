use core::ptr;

use crate::Error;

/// A table cell as the format describes it (the decoders use their own cell layouts).
#[cfg(test)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct FseEntry {
    pub base: u16,
    pub sym: u8,
    pub nb: u8,
}

/// Bytes of the header the reader may consume; every valid header is shorter.
const HEAD: usize = 504;

/// 32 bits from `bitpos` on, read in place; bytes past the end read as zero. A bit past `limit` is
/// only ever inspected by a read that also consumes it, so its value never reaches a valid result.
#[inline(always)]
fn peek_fwd(src: &[u8], bitpos: usize) -> u32 {
    let byte = bitpos >> 3;
    let w = if byte + 4 <= src.len() {
        u32::from_le(unsafe { ptr::read_unaligned(src.as_ptr().add(byte).cast::<u32>()) })
    } else {
        // Fewer than 4 bytes left from `byte`: gathered highest first, no call in the loop.
        let (mut v, mut i) = (0u32, src.len());
        while i > byte {
            i -= 1;
            v = v << 8 | unsafe { *src.get_unchecked(i) } as u32;
        }
        v
    };
    w >> (bitpos & 7)
}

/// The table log `read_ncount` leaves for a malformed header: no build accepts it (every table holds at
/// most 2^9 cells), so the build reports the header's `Error::Fse` and the reader's caller has no error
/// path of its own between the two calls.
pub(crate) const BAD_LOG: u32 = 30;

/// Reads an NCount header into `w`: the counts into `norm`, the symbol count and table log into `n` and
/// `log` (the log unchecked: the table build rejects one too large for its table, and `BAD_LOG` for a
/// malformed header), the header bytes into `used`. Returns `w.owner`, read from the block: the caller gets
/// its block back in the result register and holds nothing across the call (a returned argument would be
/// known to the compiler, which then keeps the argument instead).
#[inline(always)]
pub(crate) fn read_ncount(src: &[u8], max_sym: usize, w: &mut Scratch) -> u64 {
    // SAFETY: the pointer is the block's own, offset to its counts.
    unsafe { ncount(src, max_sym, ptr::from_mut(w).byte_add(NORM).cast::<i16>()) }
}

/// Offset of the counts in `Scratch`; `ncount` addresses the block's fields from them.
const NORM: usize = core::mem::offset_of!(Scratch, norm);

/// The field at offset `f` of the block whose counts are at `base`.
///
/// # Safety
/// As for `ncount`; `f` is the offset of a field of `Scratch`.
#[inline(always)]
unsafe fn field<T>(base: *mut i16, f: usize) -> *mut T {
    unsafe { base.byte_add(f).byte_sub(NORM).cast::<T>() }
}

/// A malformed header: `BAD_LOG` and the owner.
///
/// # Safety
/// As for `ncount`.
#[inline(always)]
unsafe fn malformed(base: *mut i16) -> u64 {
    unsafe {
        field::<u32>(base, core::mem::offset_of!(Scratch, log)).write(BAD_LOG);
        ptr::read(field::<u64>(base, core::mem::offset_of!(Scratch, owner)))
    }
}

/// The outputs are addressed from `base`, the one pointer the reader holds: with the block's address as the
/// argument, the block and its counts were two registers and the block went to a stack slot.
///
/// # Safety
/// `base` points at the `norm` field of a live `Scratch` and is derived from a pointer to the whole block.
unsafe fn ncount(src: &[u8], max_sym: usize, base: *mut i16) -> u64 {
    // Bytes from HEAD on read as zero, as if the header ended there.
    let src = &src[..src.len().min(HEAD)];
    let Some(&first) = src.first() else { return unsafe { malformed(base) } };
    let mut bitpos = 4usize;
    let mut remaining: i32 = (1 << ((first & 15) + 5)) + 1;
    let end = base.wrapping_add(max_sym.min(255) + 1);
    let mut np = base;
    while remaining > 1 {
        if np >= end {
            return unsafe { malformed(base) };
        }
        let nbits = 32 - (remaining as u32).leading_zeros();
        let v = peek_fwd(src, bitpos);
        let half = 1u32 << (nbits - 1);
        let low_count = (1u32 << nbits) - 1 - remaining as u32;
        let low = v & (half - 1);
        let value = if low < low_count {
            bitpos += nbits as usize - 1;
            low
        } else {
            let full = v & ((1u32 << nbits) - 1);
            bitpos += nbits as usize;
            if full >= half {
                full - low_count
            } else {
                full
            }
        };
        let p = value as i32 - 1;
        unsafe { *np = p as i16 };
        np = np.wrapping_add(1);
        remaining -= p.abs();
        if p == 0 {
            loop {
                let r = (peek_fwd(src, bitpos) & 3) as usize;
                bitpos += 2;
                if np.wrapping_add(r) > end {
                    return unsafe { malformed(base) };
                }
                for i in 0..3 {
                    if i < r {
                        unsafe { *np.add(i) = 0 };
                    }
                }
                np = np.wrapping_add(r);
                if r != 3 {
                    break;
                }
            }
        }
        // bitpos > 8 * len, from len alone (no second loop bound held in a register).
        if (bitpos + 7) >> 3 > src.len() {
            return unsafe { malformed(base) };
        }
    }
    if remaining != 1 {
        return unsafe { malformed(base) };
    }
    // n <= 256 (np <= end), log <= 20 (read again from the first byte where it is stored), bytes <= 504.
    // SAFETY: each field of the block `base` lies in (the caller's contract); `src` is not empty.
    unsafe {
        field::<usize>(base, core::mem::offset_of!(Scratch, n)).write((np as usize - base as usize) / 2);
        field::<u32>(base, core::mem::offset_of!(Scratch, log)).write((ptr::read(src.as_ptr()) & 15) as u32 + 5);
        field::<u64>(base, core::mem::offset_of!(Scratch, used)).write(((bitpos + 7) >> 3) as u64);
        ptr::read(field::<u64>(base, core::mem::offset_of!(Scratch, owner)))
    }
}

/// The table of `s.norm[..s.n]` at log `s.log` into `table`.
#[cfg(test)]
pub(crate) fn build_table(s: &mut Scratch, table: &mut [FseEntry]) -> Result<(), Error> {
    build_with(s, table.len(), |i, sym, nb, base| table[i] = FseEntry { base, sym, nb })
}

/// Work block of `build_with`: the normalized counts and the table log it builds from (written by the
/// caller), and its working arrays. Every byte it reads it wrote first in the same call, so a block
/// that lives with its owner is never cleared. The phases take the block alone, so the builder holds
/// nothing in registers across their calls. The arrays keep the order and offsets they had before the
/// counts joined the block (counts first measured 6 ns slower per 512-cell table).
#[repr(C)]
pub(crate) struct Scratch {
    run: [u8; 520],
    pub syms: [u8; 512],
    pub rec: [u64; 64],
    pub norm: [i16; 256],
    /// Symbols in `norm` and the table log, set by the caller.
    pub n: usize,
    pub log: u32,
    /// Nonzero when a symbol has probability -1 (an integer: the Huffman tree work block lays this
    /// block over table memory, so every bit pattern must be valid).
    low: u32,
    pub size: usize,
    high: usize,
    /// The block's own address (written by `read_ncount`'s caller and by `prepare`): what `read_ncount`
    /// and each build phase return, read from here. And the header bytes `read_ncount` read.
    pub owner: u64,
    pub used: u64,
    /// The code entries a sequence table's cell leaf reads (written by its builder).
    pub codes: u64,
    /// Rounds the block up by 64 bytes over its old size, so the sequence tables behind it in
    /// `SeqState` keep their offsets within a line (32 bytes more measured fast5 0.4% slower).
    pad: u64,
}

impl Scratch {
    #[cfg(test)]
    pub(crate) const fn new() -> Self {
        Scratch { norm: [0; 256], run: [0; 520], syms: [0; 512], rec: [0; 64], n: 0, log: 0, low: 0, size: 0, high: 0, owner: 1, used: 0, codes: 0, pad: 0 }
    }

    /// The counts read by `read_ncount` (symbols and table log) for the next build.
    #[inline(always)]
    pub(crate) fn set(&mut self, n: usize, log: u32) {
        self.n = n;
        self.log = log;
    }
}

/// Checks the symbol count and the log against the table (its cells in `size` on entry, then the size
/// at the log), writes each symbol's record, then checks the counts (none below -1, their sum the table
/// size) and writes, for probability -1, each symbol's cell at the top of the table. On null
/// (`Error::Fse`) nothing it wrote is used, so the arithmetic on an invalid distribution only wraps. The
/// checks are here, in a leaf, so the builder's caller holds no temporaries of them and has no exit
/// before its calls are done. Two passes: one pass held 13 values against the 7 volatile Win64
/// registers; each pass alone fits. Returns the block, or null for a bad table description.
#[inline(never)]
fn meta(s: &mut Scratch) -> *mut Scratch {
    let log = s.log;
    if s.n > 64 || log > 9 || 1usize << log > s.size {
        return ptr::null_mut();
    }
    s.size = 1usize << log;
    let n = s.n.min(64);
    for i in 0..n {
        // |p| is the count, and 1 for probability -1; a symbol of count 0 is never placed, so its
        // record is never read. `| 1` only gives the bit count a set bit. The log is read where used
        // (a memory operand, no register).
        let c = s.norm[i].unsigned_abs() as u32;
        let hb = 31 - (c | 1).leading_zeros();
        let log = unsafe { ptr::read(&s.log) };
        s.rec[i] = rec(c, hb, log);
    }
    // A count below -1 adds at least 2^32 - 32768 (zero-extended from 32 bits): 64 of them cannot
    // wrap the 64-bit total back to a table size, so the sum test also rejects them. The top cell
    // index stays masked to the array, exact while there are no more -1 counts than cells (every
    // valid table).
    let size = s.size;
    let (mut h, mut total) = (size, 0u64);
    for i in 0..n {
        let p = s.norm[i] as i32;
        if p == -1 {
            h = h.wrapping_sub(1) & 511;
            s.syms[h] = i as u8;
            total += 1;
        } else {
            total = total.wrapping_add(p as u32 as u64);
        }
    }
    (s.high, s.low) = (h, (h != size) as u32);
    if total != size as u64 {
        return ptr::null_mut();
    }
    me(s)
}

/// A symbol's record for count `c` (its top bit `hb`) at table log `log`, read and stepped once per cell:
/// bits 0..16 the state `x` (from `c`), bits 16..33 `x + 2^16 - 2^(hb + 1)`, whose carry into bit 32 is
/// `x >= 2^(hb + 1)`, and bits 32.. `hb - log` (wrapping). Both counters step together (`+ STEP`,
/// wrapping: at `log - hb == 1` the carry leaves bit 63 and the top field becomes 0), so `(r >> 32)` is
/// `-nb` for the cell's bit count `nb = log - hb - (x >= 2^(hb + 1))`: one shift and no compare per cell.
/// The second field stays below 2^17 (`c` <= 2^15).
#[inline(always)]
fn rec(c: u32, hb: u32, log: u32) -> u64 {
    let w = (c + 0x1_0000).wrapping_sub(2 << hb) as u64;
    c as u64 | w << 16 | (hb.wrapping_sub(log) as u64) << 32
}

/// What a cell adds to its symbol's record (both counters).
pub(crate) const STEP: u64 = 0x1_0001;

/// The cell's bit count and next-state base from its record `r` before the step, for a table of `size`.
#[inline(always)]
pub(crate) fn cell(r: u64, size: usize) -> (u32, u16) {
    let nb = ((r >> 32) as u32).wrapping_neg();
    (nb, ((r as u32).wrapping_shl(nb)).wrapping_sub(size as u32) as u16)
}

/// The block's address from `owner`. Each phase returns it read from memory, so its caller gets the
/// block back in the result register and holds nothing across the call: a function that returns its
/// argument is known to, the caller keeps the argument instead, and the callee holds it through its loops.
#[inline(always)]
fn me(s: &Scratch) -> *mut Scratch {
    // SAFETY: `s` is a valid reference.
    ptr::with_exposed_provenance_mut(unsafe { ptr::read(&s.owner) } as usize)
}

/// A byte times this fills a word with it; read where used (a memory operand: the loop's other values and
/// the block's address, live to the end, fill the volatile registers).
static BYTES: u64 = 0x0101_0101_0101_0101;

/// Returns the block.
#[inline(never)]
fn runs(s: &mut Scratch) -> *mut Scratch {
    let Scratch { norm, run, n, .. } = s;
    let mut at = run.as_mut_ptr();
    for (s, &p) in norm[..(*n).min(64)].iter().enumerate() {
        if p > 0 {
            // SAFETY: a static.
            let fill = s as u64 * unsafe { ptr::read(&BYTES) };
            let end = at.wrapping_add(p as usize);
            let mut q = at;
            loop {
                unsafe { ptr::write_unaligned(q.cast::<u64>(), fill) };
                q = q.wrapping_add(8);
                if q >= end {
                    break;
                }
            }
            at = end;
        }
    }
    me(s)
}

/// Returns the block, or null when the spread does not end where it began (a bad distribution).
#[inline(never)]
fn spread(s: &mut Scratch) -> *mut Scratch {
    let Scratch { run, syms, size, high, low, .. } = s;
    let (size, high, low) = (*size, *high, *low != 0);
    let step = (size >> 1) + (size >> 3) + 3;
    // size <= 512: the `& 511` changes nothing, but bounds `pos` for the compiler, so the cell store
    // needs no mask per cell (read from the block, the size has no known range; as an argument, its
    // range came from the callers).
    let mask = (size - 1) & 511;
    let mut pos = 0usize;
    if low {
        // high < size <= 512; the index is masked like `pos` and compared against `high` itself (the
        // run read through a pointer pair took one register more than the volatile set).
        let mut i = 0;
        while i < high {
            syms[pos & 511] = run[i & 511];
            i += 1;
            loop {
                pos = (pos + step) & mask;
                if pos < high {
                    break;
                }
            }
        }
    } else {
        for &s in &run[..size.min(512)] {
            syms[pos & 511] = s;
            pos = (pos + step) & mask;
        }
    }
    if pos != 0 {
        return ptr::null_mut();
    }
    me(s)
}

/// The phases before the cells of a table of `cells` entries, from `s.norm[..s.n]` at log `s.log`: the
/// checks, the symbol records, their runs and the spread. Returns the block from the last phase, or null
/// for a bad table description; every value used after a phase call is read back from the block, so the
/// caller holds nothing, and its one exit is after the calls.
#[inline(always)]
pub(crate) fn prepare(s: &mut Scratch, cells: usize) -> *mut Scratch {
    s.size = cells;
    s.owner = ptr::from_mut(s).expose_provenance() as u64;
    let p = meta(s);
    if p.is_null() {
        return p;
    }
    // SAFETY: each phase returns the block it was given (its `owner`, written above).
    let p = runs(unsafe { &mut *p });
    // SAFETY: as above.
    spread(unsafe { &mut *p })
}

/// Builds a table of `cells` entries from `s.norm[..s.n]` at log `s.log`, each cell through `put`. The
/// cells are read through `s` itself: a caller whose table lies in the same struct as the block (the
/// Huffman tree) holds that pointer across the phases anyway, and the returned block took a register more.
#[inline(always)]
pub(crate) fn build_with(s: &mut Scratch, cells: usize, mut put: impl FnMut(usize, u8, u8, u16)) -> Result<(), Error> {
    if prepare(s, cells).is_null() {
        return Err(Error::Fse);
    }
    let Scratch { syms, rec, size, .. } = s;
    let size = (*size).min(512);
    for (i, &s) in syms[..size].iter().enumerate() {
        let j = s as usize & 63;
        let r = rec[j];
        rec[j] = r.wrapping_add(STEP);
        let (nb, base) = cell(r, size);
        put(i, s, nb as u8, base);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seq::{LL_DEFAULT, ML_DEFAULT, OF_DEFAULT};
    use alloc::vec;
    use alloc::vec::Vec;

    /// The stepped record gives every state's bit count and base as computed from the state itself.
    #[test]
    fn cell_record_matches_state_formula() {
        for log in 0..=9u32 {
            let size = 1usize << log;
            for c in 1..=size as u32 {
                let mut r = rec(c, 31 - (c | 1).leading_zeros(), log);
                for x in c..2 * c {
                    let nb = log - (31 - x.leading_zeros());
                    assert_eq!(cell(r, size), (nb, ((x << nb) - size as u32) as u16), "log {log} count {c} state {x}");
                    r = r.wrapping_add(STEP);
                }
            }
        }
    }

    fn table(norm: &[i16], log: u32, t: &mut [FseEntry], s: &mut Scratch) -> Result<(), Error> {
        s.norm[..norm.len()].copy_from_slice(norm);
        s.set(norm.len(), log);
        build_table(s, t)
    }

    fn build(norm: &[i16], log: u32) -> Vec<FseEntry> {
        let mut t = vec![FseEntry::default(); 1 << log];
        table(norm, log, &mut t, &mut Scratch::new()).unwrap();
        t
    }

    #[test]
    fn baseline_example_probability_five() {
        let mut norm = vec![0i16; 2];
        norm[0] = 5;
        norm[1] = 123;
        let t = build(&norm, 7);
        let mine: Vec<(u8, u16)> = t.iter().filter(|e| e.sym == 0).map(|e| (e.nb, e.base)).collect();
        assert_eq!(mine, vec![(5, 32), (5, 64), (5, 96), (4, 0), (4, 16)]);
    }

    const LL_APPENDIX: [(u8, u8, u16); 64] = [
        (0, 4, 0),
        (0, 4, 16),
        (1, 5, 32),
        (3, 5, 0),
        (4, 5, 0),
        (6, 5, 0),
        (7, 5, 0),
        (9, 5, 0),
        (10, 5, 0),
        (12, 5, 0),
        (14, 6, 0),
        (16, 5, 0),
        (18, 5, 0),
        (19, 5, 0),
        (21, 5, 0),
        (22, 5, 0),
        (24, 5, 0),
        (25, 5, 32),
        (26, 5, 0),
        (27, 6, 0),
        (29, 6, 0),
        (31, 6, 0),
        (0, 4, 32),
        (1, 4, 0),
        (2, 5, 0),
        (4, 5, 32),
        (5, 5, 0),
        (7, 5, 32),
        (8, 5, 0),
        (10, 5, 32),
        (11, 5, 0),
        (13, 6, 0),
        (16, 5, 32),
        (17, 5, 0),
        (19, 5, 32),
        (20, 5, 0),
        (22, 5, 32),
        (23, 5, 0),
        (25, 4, 0),
        (25, 4, 16),
        (26, 5, 32),
        (28, 6, 0),
        (30, 6, 0),
        (0, 4, 48),
        (1, 4, 16),
        (2, 5, 32),
        (3, 5, 32),
        (5, 5, 32),
        (6, 5, 32),
        (8, 5, 32),
        (9, 5, 32),
        (11, 5, 32),
        (12, 5, 32),
        (15, 6, 0),
        (17, 5, 32),
        (18, 5, 32),
        (20, 5, 32),
        (21, 5, 32),
        (23, 5, 32),
        (24, 5, 32),
        (35, 6, 0),
        (34, 6, 0),
        (33, 6, 0),
        (32, 6, 0),
    ];

    const ML_APPENDIX: [(u8, u8, u16); 64] = [
        (0, 6, 0),
        (1, 4, 0),
        (2, 5, 32),
        (3, 5, 0),
        (5, 5, 0),
        (6, 5, 0),
        (8, 5, 0),
        (10, 6, 0),
        (13, 6, 0),
        (16, 6, 0),
        (19, 6, 0),
        (22, 6, 0),
        (25, 6, 0),
        (28, 6, 0),
        (31, 6, 0),
        (33, 6, 0),
        (35, 6, 0),
        (37, 6, 0),
        (39, 6, 0),
        (41, 6, 0),
        (43, 6, 0),
        (45, 6, 0),
        (1, 4, 16),
        (2, 4, 0),
        (3, 5, 32),
        (4, 5, 0),
        (6, 5, 32),
        (7, 5, 0),
        (9, 6, 0),
        (12, 6, 0),
        (15, 6, 0),
        (18, 6, 0),
        (21, 6, 0),
        (24, 6, 0),
        (27, 6, 0),
        (30, 6, 0),
        (32, 6, 0),
        (34, 6, 0),
        (36, 6, 0),
        (38, 6, 0),
        (40, 6, 0),
        (42, 6, 0),
        (44, 6, 0),
        (1, 4, 32),
        (1, 4, 48),
        (2, 4, 16),
        (4, 5, 32),
        (5, 5, 32),
        (7, 5, 32),
        (8, 5, 32),
        (11, 6, 0),
        (14, 6, 0),
        (17, 6, 0),
        (20, 6, 0),
        (23, 6, 0),
        (26, 6, 0),
        (29, 6, 0),
        (52, 6, 0),
        (51, 6, 0),
        (50, 6, 0),
        (49, 6, 0),
        (48, 6, 0),
        (47, 6, 0),
        (46, 6, 0),
    ];

    const OF_APPENDIX: [(u8, u8, u16); 32] = [
        (0, 5, 0),
        (6, 4, 0),
        (9, 5, 0),
        (15, 5, 0),
        (21, 5, 0),
        (3, 5, 0),
        (7, 4, 0),
        (12, 5, 0),
        (18, 5, 0),
        (23, 5, 0),
        (5, 5, 0),
        (8, 4, 0),
        (14, 5, 0),
        (20, 5, 0),
        (2, 5, 0),
        (7, 4, 16),
        (11, 5, 0),
        (17, 5, 0),
        (22, 5, 0),
        (4, 5, 0),
        (8, 4, 16),
        (13, 5, 0),
        (19, 5, 0),
        (1, 5, 0),
        (6, 4, 16),
        (10, 5, 0),
        (16, 5, 0),
        (28, 5, 0),
        (27, 5, 0),
        (26, 5, 0),
        (25, 5, 0),
        (24, 5, 0),
    ];

    fn check(norm: &[i16], log: u32, want: &[(u8, u8, u16)]) {
        let t = build(norm, log);
        for (i, (e, w)) in t.iter().zip(want).enumerate() {
            assert_eq!((e.sym, e.nb, e.base), *w, "state {i}");
        }
    }

    #[test]
    fn predefined_tables_match_appendix() {
        check(&LL_DEFAULT, 6, &LL_APPENDIX);
        check(&ML_DEFAULT, 6, &ML_APPENDIX);
        check(&OF_DEFAULT, 5, &OF_APPENDIX);
    }

    fn write_bits(out: &mut Vec<u8>, acc: &mut u64, n: &mut u32, v: u32, bits: u32) {
        *acc |= (v as u64) << *n;
        *n += bits;
        while *n >= 8 {
            out.push(*acc as u8);
            *acc >>= 8;
            *n -= 8;
        }
    }

    #[test]
    fn ncount_round_trip() {
        let norm: [i16; 4] = [20, -1, 0, 11];
        let mut out = Vec::new();
        let mut acc = 0u64;
        let mut n = 0u32;
        write_bits(&mut out, &mut acc, &mut n, 0, 4);
        let mut remaining: i32 = 33;
        let mut i = 0;
        while remaining > 1 {
            let p = norm[i];
            let value = (p + 1) as u32;
            let nbits = 32 - (remaining as u32).leading_zeros();
            let low_count = (1u32 << nbits) - 1 - remaining as u32;
            if value < low_count {
                write_bits(&mut out, &mut acc, &mut n, value, nbits - 1);
            } else if value < (1 << (nbits - 1)) {
                write_bits(&mut out, &mut acc, &mut n, value, nbits);
            } else {
                write_bits(&mut out, &mut acc, &mut n, value + low_count, nbits);
            }
            remaining -= p.abs() as i32;
            if p == 0 {
                write_bits(&mut out, &mut acc, &mut n, 0, 2);
            }
            i += 1;
        }
        if n > 0 {
            out.push(acc as u8);
        }
        let mut s = Scratch::new();
        assert_eq!(read_ncount(&out, 255, &mut s), 1);
        assert_eq!((s.n, s.log, s.used as usize), (4, 5, out.len()));
        assert_eq!(&s.norm[..4], &norm);
        // The table log is checked by the build: log 5 needs 32 cells.
        assert_eq!(build_table(&mut s, &mut [FseEntry::default(); 16]), Err(Error::Fse));
        // The header followed by more data: read in place, same result.
        let mut more = out.clone();
        more.extend_from_slice(&[0xFF; 9]);
        let mut s = Scratch::new();
        assert_eq!(read_ncount(&more, 255, &mut s), 1);
        assert_eq!((s.n, s.log, s.used as usize), (4, 5, out.len()));
        assert_eq!(&s.norm[..4], &norm);
    }

    #[test]
    fn rejects_bad_distributions() {
        let mut t = vec![FseEntry::default(); 64];
        let mut s = Scratch::new();
        assert!(table(&[30, 1], 5, &mut t, &mut s).is_err());
        assert!(table(&[33], 5, &mut t, &mut s).is_err());
        let mut fails = |src: &[u8], s: &mut Scratch| {
            read_ncount(src, 255, s);
            s.log == BAD_LOG && build_table(s, &mut t).is_err()
        };
        assert!(fails(&[0x0F], &mut s));
        assert!(fails(&[], &mut s));
        assert!(fails(&[0x00], &mut s));
    }
}
