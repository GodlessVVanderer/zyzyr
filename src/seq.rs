use alloc::boxed::Box;
use core::hint::select_unpredictable;
use core::ptr;

use crate::bits::peek;
use crate::cpu::Cpu;
use crate::fse::{prepare, read_ncount, Scratch, STEP};
use crate::{at, put, Error, Job};

pub(crate) const LL_DEFAULT: [i16; 36] = [4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1];

pub(crate) const ML_DEFAULT: [i16; 53] =
    [1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1];

pub(crate) const OF_DEFAULT: [i16; 29] = [1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1];

const LL_BASE: [u32; 36] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];

const LL_BITS: [u8; 36] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

const ML_BASE: [u32; 53] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051,
    4099, 8195, 16387, 32771, 65539,
];

const ML_BITS: [u8; 53] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

#[derive(Clone, Copy, Default)]
#[repr(C, align(8))]
pub(crate) struct Val {
    base: u32,
    emask: u32,
}

#[derive(Clone, Copy, Default)]
#[repr(C, align(8))]
pub(crate) struct Step {
    nmask: u32,
    next: u16,
    nextra: i8,
    nnb: i8,
}

impl Step {
    #[inline(always)]
    fn extra(&self) -> u32 {
        (-(self.nextra as i32)) as u32
    }

    #[inline(always)]
    fn nb(&self) -> u32 {
        (-(self.nnb as i32)) as u32
    }

    #[inline(always)]
    fn next(&self) -> usize {
        self.next as usize
    }
}

impl Step {
    /// The entry from its fields as one 8-byte value, so a table write is one store.
    #[inline(always)]
    fn pack(nmask: u32, next: u16, nextra: i8, nnb: i8) -> Step {
        let (m, n, x, b) = (nmask as u64, next as u64, nextra as u8 as u64, nnb as u8 as u64);
        let w = if cfg!(target_endian = "little") { m | n << 32 | x << 48 | b << 56 } else { m << 32 | n << 16 | x << 8 | b };
        // SAFETY: `Step` is 8 bytes of integers in `repr(C)` order, no padding; every bit pattern is valid.
        unsafe { core::mem::transmute::<u64, Step>(w) }
    }
}

/// Value entry and `-extra` of every code of one kind, by symbol; built at compile time. Symbols past
/// the kind's last are never placed in a table (`read_ncount` stops at the kind's symbol count).
struct Codes {
    val: [Val; 64],
    nextra: [i8; 64],
}

impl Codes {
    const fn of(kind: Kind) -> Codes {
        let mut c = Codes { val: [Val { base: 0, emask: 0 }; 64], nextra: [0; 64] };
        let mut s = 0;
        while s <= kind.limits().0 {
            let (base, extra) = kind.code(s as u8);
            let extra = extra & 31;
            c.val[s] = Val { base, emask: ((1u64 << extra) - 1) as u32 };
            c.nextra[s] = -(extra as i8);
            s += 1;
        }
        c
    }
}

static LL_CODES: Codes = Codes::of(Kind::Ll);
static OF_CODES: Codes = Codes::of(Kind::Of);
static ML_CODES: Codes = Codes::of(Kind::Ml);

#[inline(always)]
fn entry(base: u32, extra: u8, nb: u8, next: u16) -> (Val, Step) {
    let (extra, nb) = (extra & 31, nb & 15);
    (Val { base, emask: ((1u64 << extra) - 1) as u32 }, Step { nmask: (1u32 << nb) - 1, next, nextra: -(extra as i8), nnb: -(nb as i8) })
}

#[inline(always)]
fn low(v: u64, n: u32) -> usize {
    (v & ((1u64 << n) - 1)) as usize
}

#[inline(always)]
fn lane(x: u64, shift: i8, mask: u32) -> u32 {
    x.wrapping_shr(shift as u8 as u32) as u32 & mask
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Ll,
    Of,
    Ml,
}

impl Kind {
    const fn limits(self) -> (usize, u32) {
        match self {
            Kind::Ll => (35, 9),
            Kind::Of => (31, 8),
            Kind::Ml => (52, 9),
        }
    }

    const fn code(self, sym: u8) -> (u32, u8) {
        let s = sym as usize;
        match self {
            Kind::Ll => (LL_BASE[s], LL_BITS[s]),
            Kind::Of => (1u32 << s, sym),
            Kind::Ml => (ML_BASE[s], ML_BITS[s]),
        }
    }

    fn codes(self) -> &'static Codes {
        match self {
            Kind::Ll => &LL_CODES,
            Kind::Of => &OF_CODES,
            Kind::Ml => &ML_CODES,
        }
    }
}

pub(crate) struct SeqTable<const N: usize> {
    pub val: [Val; N],
    pub step: [Step; N],
    pub log: u32,
    pub valid: bool,
}

impl<const N: usize> SeqTable<N> {
    fn rle(&mut self, sym: u8, kind: Kind) {
        let (base, extra) = kind.code(sym);
        (self.val[0], self.step[0]) = entry(base, extra, 0, 0);
        self.log = 0;
        self.valid = true;
    }

    /// Copies inline, 16 bytes (two entries) at a time; a one-entry table copies two (N >= 256). The count
    /// is bounded by `N` (a table's log never passes it): the bound check's panic call made `setup` keep a
    /// frame on aarch64 on every path.
    fn copy_from(&mut self, other: &SeqTable<N>) {
        let n = (1usize << other.log).clamp(2, N);
        let (sv, ss) = (other.val[..n].as_ptr().cast::<u128>(), other.step[..n].as_ptr().cast::<u128>());
        let (dv, ds) = (self.val[..n].as_mut_ptr().cast::<u128>(), self.step[..n].as_mut_ptr().cast::<u128>());
        for i in 0..n / 2 {
            // SAFETY: i < n / 2, so 16-byte pair i lies inside the first n entries of each array.
            unsafe {
                ptr::write_unaligned(dv.add(i), ptr::read_unaligned(sv.add(i)));
                ptr::write_unaligned(ds.add(i), ptr::read_unaligned(ss.add(i)));
            }
        }
        self.log = other.log;
        self.valid = true;
    }
}

/// Offsets from the state's work block of its six tables: `cells` addresses the table it builds from the
/// block's pointer (one register fewer through the cell loop).
const fn table_at(table: usize) -> usize {
    table - core::mem::offset_of!(SeqState, work)
}
const LL_AT: usize = table_at(core::mem::offset_of!(SeqState, ll));
const OF_AT: usize = table_at(core::mem::offset_of!(SeqState, of));
const ML_AT: usize = table_at(core::mem::offset_of!(SeqState, ml));
const LL_PRE_AT: usize = table_at(core::mem::offset_of!(SeqState, ll_pre));
const OF_PRE_AT: usize = table_at(core::mem::offset_of!(SeqState, of_pre));
const ML_PRE_AT: usize = table_at(core::mem::offset_of!(SeqState, ml_pre));

/// The table of `N` cells at offset `AT` from the state's work block, of kind `kind`, from the counts and
/// log in the block (its FSE build arrays too), so nothing is cleared per table. The kind's code entries
/// go into the block, the phases hand the block back, and the cells are the tail call on every path:
/// nothing is held across the calls and there is one exit (an error exit before the tail call kept a
/// second epilogue on aarch64). Returns the header bytes `read_ncount` left in the block (what `setup`
/// returns, so `setup` ends in this call), or `BAD_FSE`; a table log above the table's is rejected.
#[inline(always)]
fn build<const N: usize, const AT: usize>(st: &mut SeqState, kind: Kind) -> u32 {
    // The whole state's provenance, for the table `cells` addresses from the block.
    ptr::from_mut(st).expose_provenance();
    put(&mut st.work.codes, ptr::from_ref(kind.codes()).expose_provenance() as u64);
    cells::<N, AT>(prepare(&mut st.work, N))
}

/// The cells of the table at `AT` from the block, from the spread symbols and their records, then its
/// log and validity; returns the header bytes, or `BAD_FSE` for a null block (a bad description). A
/// leaf in the 7 volatile Win64 registers: the code entries are read from the block where used, the table
/// is addressed from the block, and the code entries are read before the record so that their address is
/// dead by the time the step is packed.
#[inline(never)]
fn cells<const N: usize, const AT: usize>(s: *mut Scratch) -> u32 {
    if s.is_null() {
        return BAD_FSE;
    }
    // SAFETY: a non-null block is the one `prepare` was given, the work block of a live state that `build`
    // exposed; the table at `AT` from it is that state's, live and not otherwise borrowed.
    let (s, t) = unsafe { (&mut *s, &mut *ptr::with_exposed_provenance_mut::<SeqTable<N>>(s.addr() + AT)) };
    t.valid = false;
    let size = s.size.min(N);
    for i in 0..size {
        // SAFETY: i < size <= N (`meta` checked the log against N); j < 64, inside `rec` and the code
        // arrays; the code entries are one of the three statics, written by `build`.
        unsafe {
            let j = *s.syms.get_unchecked(i) as usize & 63;
            let r = *s.rec.get_unchecked(j);
            *s.rec.get_unchecked_mut(j) = r.wrapping_add(STEP);
            // A placed symbol's state is never 0. The test keeps the count's store next to its load:
            // without a branch there it sank to the end of the iteration (5.8% slower per 512 cells).
            if r as u16 == 0 {
                return bad_fse();
            }
            let c = &*ptr::with_exposed_provenance::<Codes>(at(&s.codes) as usize);
            *t.val.get_unchecked_mut(i) = *c.val.get_unchecked(j);
            let nextra = *c.nextra.get_unchecked(j);
            // The record's top field is `-nb` (`nb <= 9`): rotating right by it is rotating left by `nb`, so
            // the state's shift and the mask take it as it is and no negation is held beside it. The state
            // shifted left by nb stays below 2^19 (no bit wraps); `1 ror 0` - 1 is the mask 0 of nb = 0.
            let nnb = (r >> 32) as u32;
            // The size's low word read where used, so it folds into the subtract (a register held it).
            let low = ptr::from_ref(&s.size).cast::<u32>().add(if cfg!(target_endian = "little") { 0 } else { 1 });
            let next = (r as u16 as u32).rotate_right(nnb).wrapping_sub(ptr::read(low)) as u16;
            *t.step.get_unchecked_mut(i) = Step::pack(1u32.rotate_right(nnb).wrapping_sub(1), next, nextra, nnb as i8);
        }
    }
    t.log = at(&s.log);
    t.valid = true;
    at(&s.used) as u32
}

/// `BAD_FSE` read from memory where returned: as a constant (inline or from a leaf, whose result the
/// compiler propagates), it waited in a callee-saved register through the cell loop.
static FSE_BAD: u32 = BAD_FSE;

#[inline(always)]
fn bad_fse() -> u32 {
    // SAFETY: a static.
    unsafe { ptr::read(&FSE_BAD) }
}

/// The work blocks of `execute` and `setup` come first, so their fields lie at small offsets from the
/// state pointer (aarch64 load offsets reach 32 KB; the tables take 40 KB).
#[repr(C)]
pub(crate) struct SeqState {
    span: Span,
    /// The first 8 bytes of a sequence stream (where it ends, it is read backward) above 8 zero bytes.
    edge: [u8; 16],
    pub reps: [u32; 3],
    head: Head,
    /// Work block of `setup`: the normalized counts of a table description and the FSE build arrays.
    work: Scratch,
    pub ll: SeqTable<512>,
    pub of: SeqTable<256>,
    pub ml: SeqTable<512>,
    ll_pre: SeqTable<512>,
    of_pre: SeqTable<256>,
    ml_pre: SeqTable<512>,
}

impl SeqState {
    /// Built in place in its box: the allocation comes zeroed (every table, the work blocks and the
    /// edge copy start as zero), so only the driver block's error and the repeat offsets are written, and the
    /// predefined tables are built where they live. Built as a value, the 45 KB state went through the
    /// stack into the box. Not inlined, so the decoder constructors hold nothing across it. The builds
    /// take the raw state: with the box alive across them, its address went to a stack slot for the
    /// unwind path that frees it.
    #[inline(never)]
    pub(crate) fn boxed() -> Box<SeqState> {
        let p = Box::into_raw(Box::<SeqState>::new_zeroed()).cast::<SeqState>();
        // SAFETY: `p` is the box's allocation. Every field but `span.err` is integers, arrays of
        // integers and `bool`s, for which zero bytes are valid; the `Option<Error>` is written before
        // the state is used, and the box is rebuilt from the pointer it came from.
        unsafe {
            (&raw mut (*p).span.err).write(None);
            (&raw mut (*p).reps).write([1, 4, 8]);
            predefined::<LL>(&mut *p);
            predefined::<OF>(&mut *p);
            predefined::<ML>(&mut *p);
            Box::from_raw(p)
        }
    }

    /// A dictionary's state, built in place as `boxed` builds a decoder's but without the predefined tables:
    /// a dictionary's tables are all FSE coded (`dict` calls `setup` in mode 2 only) and `load` copies only
    /// the three tables and the repeat offsets, so the predefined ones are never read. Its own code (no call
    /// of `boxed` or `predefined`), so the decoder's constructor is unchanged.
    #[inline(always)]
    pub(crate) fn for_dict() -> Box<SeqState> {
        let p = Box::into_raw(Box::<SeqState>::new_zeroed()).cast::<SeqState>();
        // SAFETY: as in `boxed`; the repeat offsets are written by `dict` before the state is used.
        unsafe {
            (&raw mut (*p).span.err).write(None);
            Box::from_raw(p)
        }
    }

    pub(crate) fn reset(&mut self) {
        self.ll.valid = false;
        self.of.valid = false;
        self.ml.valid = false;
        self.reps = [1, 4, 8];
    }

    /// Inlined into each of the context's resets (the frame code's, out of line, and the stream decoder's,
    /// inlined there): shared out of line, the stream decoder's frame start made one more call.
    #[inline(always)]
    pub(crate) fn load(&mut self, d: &SeqState) {
        self.ll.copy_from(&d.ll);
        self.of.copy_from(&d.of);
        self.ml.copy_from(&d.ml);
        self.reps = d.reps;
    }

    /// Binds a formatted dictionary after its magic number and id (the Huffman tree first) to the section
    /// reader for `dict`: its sequence tables start at `p`.
    pub(crate) fn bind_dict(&mut self, src: &[u8], p: usize) {
        put(&mut self.head.src, src.as_ptr() as usize);
        put(&mut self.head.len, src.len());
        put(&mut self.head.p, p);
    }

    /// What `bind_dict` bound, read from the block.
    #[inline(always)]
    pub(crate) fn dict_input<'a>(&self) -> &'a [u8] {
        // SAFETY: bound by `bind_dict` to a live dictionary.
        unsafe { core::slice::from_raw_parts(at(&self.head.src) as *const u8, at(&self.head.len)) }
    }

    /// The bytes after what `dict` has read (the dictionary's content), read from the block. Only after
    /// `dict` succeeded, which leaves `p <= len`.
    #[inline(always)]
    pub(crate) fn dict_rest<'a>(&self) -> &'a [u8] {
        let (src, len, p) = (at(&self.head.src), at(&self.head.len), at(&self.head.p));
        // SAFETY: bound by `bind_dict` to a live dictionary; `p <= len` (above).
        unsafe { core::slice::from_raw_parts((src + p) as *const u8, len - p) }
    }

    /// The reader `dict` replaced, for the equivalence test: the three tables from `src`, the bytes read.
    #[cfg(test)]
    pub(crate) fn read_dict_tables(&mut self, src: &[u8]) -> Result<usize, Error> {
        let mut p = 0usize;
        let rest = |p: usize| src.get(p..).ok_or(Error::Truncated);
        p += read(setup::<OF>(self, 2, rest(p)?))? as usize;
        p += read(setup::<ML>(self, 2, rest(p)?))? as usize;
        p += read(setup::<LL>(self, 2, rest(p)?))? as usize;
        Ok(p)
    }

    /// Whether the repeat offsets and the three tables (log, validity and entries) are the same.
    #[cfg(test)]
    pub(crate) fn same_tables(&self, o: &SeqState) -> bool {
        fn same<const N: usize>(a: &SeqTable<N>, b: &SeqTable<N>) -> bool {
            let n = 1usize << a.log;
            // SAFETY: `Val` and `Step` are 8 bytes without padding (`repr(C, align(8))`, two `u32` and
            // `u32, u16, i8, i8`), so each entry reads as one `u64`.
            let w = |v: &[Val], s: &[Step]| unsafe { (v.as_ptr().cast::<u64>().read(), s.as_ptr().cast::<u64>().read()) };
            a.log == b.log && a.valid == b.valid && (0..n).all(|i| w(&a.val[i..], &a.step[i..]) == w(&b.val[i..], &b.step[i..]))
        }
        self.reps == o.reps && same(&self.ll, &o.ll) && same(&self.of, &o.of) && same(&self.ml, &o.ml)
    }
}

/// Reads a formatted dictionary's three tables (offsets, match lengths, literal lengths, each FSE coded)
/// and its repeat offsets through the section reader `bind_dict` set, as `parse_header` reads a block's,
/// so the caller holds nothing across the calls: the state's address comes from `st` at each step (the
/// dictionary build reads it from the dictionary; a leaf of its own kept a frame of its own on aarch64).
/// The offsets are checked against the content's length. False on a malformed dictionary.
#[inline(always)]
pub(crate) fn dict<'a>(st: impl Fn() -> &'a mut SeqState) -> bool {
    let s = st();
    let Ok(n) = read(setup::<OF>(s, 2, s.rest())) else { return false };
    st().head.p += n as usize;
    let s = st();
    let Ok(n) = read(setup::<ML>(s, 2, s.rest())) else { return false };
    st().head.p += n as usize;
    let s = st();
    let Ok(n) = read(setup::<LL>(s, 2, s.rest())) else { return false };
    let st = st();
    st.head.p += n as usize;
    let rest = st.rest();
    let Some(r) = rest.first_chunk::<12>() else { return false };
    let content = rest.len() - 12;
    for i in 0..3 {
        let v = u32::from_le_bytes([r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]]);
        if v == 0 || v as usize > content {
            return false;
        }
        st.reps[i] = v;
    }
    st.head.p += 12;
    true
}

const LL: u8 = 0;
const OF: u8 = 1;
const ML: u8 = 2;

/// The predefined table of kind `K` from its counts, built in the state's work block; the state is the
/// only argument (inlined three times into `SeqState::new`, the copy and build grew its stack moves).
#[inline(never)]
fn predefined<const K: u8>(st: &mut SeqState) {
    let (norm, log): (&[i16], u32) = match K {
        LL => (&LL_DEFAULT, 6),
        OF => (&OF_DEFAULT, 5),
        _ => (&ML_DEFAULT, 6),
    };
    let w = &mut st.work;
    w.norm[..norm.len()].copy_from_slice(norm);
    w.set(norm.len(), log);
    let r = match K {
        LL => build::<512, LL_PRE_AT>(st, Kind::Ll),
        OF => build::<256, OF_PRE_AT>(st, Kind::Of),
        _ => build::<512, ML_PRE_AT>(st, Kind::Ml),
    };
    assert!(r < BAD_TRUNC);
}

/// What `setup` returns for each error, above every byte count (a header is at most 504 bytes).
const BAD_TRUNC: u32 = u32::MAX - 2;
const BAD_SEQ: u32 = u32::MAX - 1;
const BAD_FSE: u32 = u32::MAX;

/// A `setup` result as the bytes read or its error.
#[inline(always)]
fn read(n: u32) -> Result<u32, Error> {
    if n < BAD_TRUNC {
        Ok(n)
    } else if n == BAD_TRUNC {
        Err(Error::Truncated)
    } else if n == BAD_SEQ {
        Err(Error::Sequences)
    } else {
        Err(Error::Fse)
    }
}

/// Sets up the table of kind `K` (`LL`, `OF`, `ML`) in mode `mode` from `src`; returns the bytes read or
/// a `BAD_*` code, one word in a register on every target (a `Result` came back through memory on
/// wasm32). Holds nothing across its calls: `read_ncount` hands the state back from the block and the
/// table build is the last call.
#[inline(never)]
fn setup<const K: u8>(st: &mut SeqState, mode: u8, src: &[u8]) -> u32 {
    if mode == 2 {
        return fse::<K>(st, src);
    }
    match K {
        LL => table(&mut st.ll, &st.ll_pre, mode, src, Kind::Ll),
        OF => table(&mut st.of, &st.of_pre, mode, src, Kind::Of),
        _ => table(&mut st.ml, &st.ml_pre, mode, src, Kind::Ml),
    }
}

/// Mode 2: the table from the FSE description at `src`. On aarch64 out of line, tail-called: inlined, its
/// call made `setup` keep a frame record on every path (the frame was not shrink-wrapped to this one).
/// Elsewhere inlined (0 stack moves either way): one function fewer per table on the block path.
#[cfg_attr(target_arch = "aarch64", inline(never))]
#[cfg_attr(not(target_arch = "aarch64"), inline(always))]
fn fse<const K: u8>(st: &mut SeqState, src: &[u8]) -> u32 {
    let max_sym = match K {
        LL => Kind::Ll,
        OF => Kind::Of,
        _ => Kind::Ml,
    }
    .limits()
    .0;
    let me = (ptr::from_mut(st).expose_provenance() + core::mem::offset_of!(SeqState, work)) as u64;
    put(&mut st.work.owner, me);
    // A malformed header leaves `BAD_LOG`, which the build rejects.
    let p = read_ncount(src, max_sym, &mut st.work);
    // SAFETY: `p` is the address of the state's work block, written to the block above and read back by
    // `read_ncount`; the state is the struct that holds it.
    let st = unsafe { &mut *ptr::with_exposed_provenance_mut::<SeqState>(p as usize - core::mem::offset_of!(SeqState, work)) };
    match K {
        LL => build::<512, LL_AT>(st, Kind::Ll),
        OF => build::<256, OF_AT>(st, Kind::Of),
        _ => build::<512, ML_AT>(st, Kind::Ml),
    }
}

/// Modes 0, 1 and 3 (predefined, RLE, repeat): no calls.
#[inline(always)]
fn table<const N: usize>(dst: &mut SeqTable<N>, pre: &SeqTable<N>, mode: u8, src: &[u8], kind: Kind) -> u32 {
    match mode {
        0 => {
            dst.copy_from(pre);
            0
        }
        1 => {
            let Some(&s) = src.first() else { return BAD_TRUNC };
            if s as usize > kind.limits().0 {
                return BAD_SEQ;
            }
            dst.rle(s, kind);
            1
        }
        _ => {
            if dst.valid {
                0
            } else {
                BAD_SEQ
            }
        }
    }
}

/// What `parse_header` read, kept in the state: the section, the bytes read of it, the table modes and
/// the sequence count. Each `setup` call reads its source from here, so the caller holds nothing
/// across the calls.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct Head {
    src: usize,
    len: usize,
    p: usize,
    modes: usize,
    nb: usize,
}

/// Reads the sequences section header and sets up the three tables; the count and the stream that
/// follows are then `count` and `stream`. False with the error left for `error`. Inlined: as its own
/// function it held only the state across the `setup` calls, but the call per block lost w10 time.
#[inline(always)]
pub(crate) fn parse_header(st: &mut SeqState, src: &[u8]) -> bool {
    match header(st, src) {
        Ok(()) => true,
        Err(e) => {
            st.span.err = Some(e);
            false
        }
    }
}

#[inline(always)]
fn header(st: &mut SeqState, src: &[u8]) -> Result<(), Error> {
    let b0 = *src.first().ok_or(Error::Truncated)? as usize;
    let byte = |i: usize| src.get(i).map(|&b| b as usize).ok_or(Error::Truncated);
    let (nb, p) = if b0 == 0 {
        st.head = Head { src: src.as_ptr() as usize, len: src.len(), p: 1, modes: 0, nb: 0 };
        return Ok(());
    } else if b0 < 128 {
        (b0, 1)
    } else if b0 < 255 {
        (((b0 - 128) << 8) + byte(1)?, 2)
    } else {
        (byte(1)? + (byte(2)? << 8) + 0x7F00, 3)
    };
    let modes = byte(p)?;
    if modes & 3 != 0 {
        return Err(Error::Sequences);
    }
    st.head = Head { src: src.as_ptr() as usize, len: src.len(), p: p + 1, modes, nb };
    st.head.p += read(setup::<LL>(st, (modes >> 6) as u8, st.rest()))? as usize;
    st.head.p += read(setup::<OF>(st, (st.head.modes >> 4) as u8 & 3, st.rest()))? as usize;
    st.head.p += read(setup::<ML>(st, (st.head.modes >> 2) as u8 & 3, st.rest()))? as usize;
    Ok(())
}

impl SeqState {
    /// The section after what `parse_header` has read of it (`setup` never reports more bytes than its
    /// source holds; the bound only keeps the slice valid).
    #[inline(always)]
    fn rest<'a>(&self) -> &'a [u8] {
        let h = self.head;
        // SAFETY: `head` was bound to a live section by `parse_header` for this call.
        unsafe { core::slice::from_raw_parts((h.src + h.p.min(h.len)) as *const u8, h.len - h.p.min(h.len)) }
    }

    /// The error of a `parse_header` that returned false.
    #[inline(always)]
    pub(crate) fn error(&self) -> Error {
        self.span.err.unwrap_or(Error::Sequences)
    }

    /// The sequence count of the section `parse_header` read.
    #[inline(always)]
    pub(crate) fn count(&self) -> usize {
        self.head.nb
    }

    /// The sequence stream after the section header and tables.
    #[inline(always)]
    pub(crate) fn stream<'a>(&self) -> &'a [u8] {
        self.rest()
    }
}

/// `dict_match` with 16-byte pieces.
#[cold]
#[inline(never)]
unsafe fn dict_plain(sp: &Span, o: *mut u8, offset: usize, ml: usize) -> DictOut {
    unsafe { dict_match::<Narrow>(sp, o, offset, ml) }
}

/// `dict_match` with 32-byte pieces.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[cold]
#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn dict_wide(sp: &Span, o: *mut u8, offset: usize, ml: usize) -> DictOut {
    unsafe { dict_match::<Wide>(sp, o, offset, ml) }
}

/// Result of the `dict_match` leaves. A `Result` where it comes back in a register: the x86-64 caller
/// keeps its own error codes in the register the callee's `Result` arrives in, and any other type
/// changed the caller's register allocation. A flag on wasm32, where a `Result` from these leaves came
/// back through a stack slot of the caller.
#[cfg(not(target_arch = "wasm32"))]
type DictOut = Result<(), Error>;
#[cfg(target_arch = "wasm32")]
type DictOut = bool;

#[cfg(not(target_arch = "wasm32"))]
const DICT_OK: DictOut = Ok(());
#[cfg(not(target_arch = "wasm32"))]
const DICT_BAD: DictOut = Err(Error::Offset);
#[cfg(target_arch = "wasm32")]
const DICT_OK: DictOut = true;
#[cfg(target_arch = "wasm32")]
const DICT_BAD: DictOut = false;

/// `dict_match`'s result as the caller's `Result`.
#[inline(always)]
fn dict_result(r: DictOut) -> Result<(), Error> {
    #[cfg(not(target_arch = "wasm32"))]
    return r;
    #[cfg(target_arch = "wasm32")]
    return if r { Ok(()) } else { Err(Error::Offset) };
}

/// A match that reaches below the frame start into the dictionary; `o` is where it is written. A leaf:
/// the dictionary part is copied without a call (through `memcpy` the output, the length, the frame
/// start and both counts waited across it in callee-saved registers), and the frame start is read
/// from the block after it. `DICT_BAD` for an offset past the window or the dictionary.
#[inline(always)]
unsafe fn dict_match<M: Mover>(sp: &Span, o: *mut u8, offset: usize, ml: usize) -> DictOut {
    let have = o as usize - sp.start;
    let reach = if sp.ring { offset } else { have } > sp.window;
    // Tested on `back` alone (`offset > have + dict_len`), so `have` and `offset` end here.
    let back = offset.wrapping_sub(have);
    if offset <= have || reach || back > sp.dict_len {
        return DICT_BAD;
    }
    let n = back.min(ml);
    unsafe {
        copy_free::<M>((sp.dict + sp.dict_len - back) as *const u8, o, n);
        // The length (the sequence's own, `run_impl` passed it from there) and the frame start again
        // from the block: neither the length nor `back` waits through the copy.
        let (fs, ml) = (at(&sp.start) as *const u8, at(&sp.seq[1]) as usize);
        for i in n..ml {
            *o.add(i) = *fs.add(i - n);
        }
    }
    DICT_OK
}

/// Copies `n` bytes from `src` to `dst` without a call: the first 64 bytes, then 128 bytes (four
/// 32-byte pieces, `M`) per step at cache-line aligned destinations (an unaligned 32-byte store
/// crosses a line every second piece), then up to 64 bytes and the 64 bytes ending at `n` (pieces may
/// overlap); below 64 bytes two pieces of the widest width that fits.
///
/// # Safety
/// As `ptr::copy_nonoverlapping(src, dst, n)`.
#[inline(always)]
unsafe fn copy_free<M: Mover>(src: *const u8, dst: *mut u8, n: usize) {
    // SAFETY: by the caller; every access lies in the first `n` bytes of `src` or `dst`.
    unsafe {
        if n >= 64 {
            let last = n - 64;
            M::copy32(dst, src);
            M::copy32(dst.add(32), src.add(32));
            let mut i = 64 - (dst as usize & 63);
            while i + 64 < last {
                M::copy32(dst.add(i), src.add(i));
                M::copy32(dst.add(i + 32), src.add(i + 32));
                M::copy32(dst.add(i + 64), src.add(i + 64));
                M::copy32(dst.add(i + 96), src.add(i + 96));
                i += 128;
            }
            if i < last {
                M::copy32(dst.add(i), src.add(i));
                M::copy32(dst.add(i + 32), src.add(i + 32));
            }
            M::copy32(dst.add(last), src.add(last));
            M::copy32(dst.add(last + 32), src.add(last + 32));
        } else if n >= 32 {
            M::copy32(dst, src);
            M::copy32(dst.add(n - 32), src.add(n - 32));
        } else if n >= 16 {
            let (a, b) = (ptr::read_unaligned(src as *const u128), ptr::read_unaligned(src.add(n - 16) as *const u128));
            ptr::write_unaligned(dst as *mut u128, a);
            ptr::write_unaligned(dst.add(n - 16) as *mut u128, b);
        } else if n >= 8 {
            let (a, b) = (ptr::read_unaligned(src as *const u64), ptr::read_unaligned(src.add(n - 8) as *const u64));
            ptr::write_unaligned(dst as *mut u64, a);
            ptr::write_unaligned(dst.add(n - 8) as *mut u64, b);
        } else if n >= 4 {
            let (a, b) = (ptr::read_unaligned(src as *const u32), ptr::read_unaligned(src.add(n - 4) as *const u32));
            ptr::write_unaligned(dst as *mut u32, a);
            ptr::write_unaligned(dst.add(n - 4) as *mut u32, b);
        } else if n >= 2 {
            let (a, b) = (ptr::read_unaligned(src as *const u16), ptr::read_unaligned(src.add(n - 2) as *const u16));
            ptr::write_unaligned(dst as *mut u16, a);
            ptr::write_unaligned(dst.add(n - 2) as *mut u16, b);
        } else if n == 1 {
            *dst = *src;
        }
    }
}

const DIST: [usize; 16] = [8, 8, 8, 9, 8, 10, 12, 14, 8, 9, 10, 11, 12, 13, 14, 15];

#[inline(always)]
unsafe fn wild16(dst: *mut u8, src: *const u8, len: usize) {
    unsafe {
        let end = dst.wrapping_add(len);
        let (mut d, mut s) = (dst, src);
        loop {
            ptr::copy_nonoverlapping(s, d, 16);
            d = d.wrapping_add(16);
            s = s.wrapping_add(16);
            if d >= end {
                break;
            }
        }
    }
}

#[inline(always)]
unsafe fn spread<M: Mover>(mut d: *mut u8, mut s: *const u8, end: *mut u8) {
    unsafe {
        while d < end {
            M::copy32(d, s);
            d = d.wrapping_add(32);
            s = s.wrapping_add(32);
        }
    }
}

#[inline(always)]
unsafe fn literals<M: Mover>(o: *mut u8, l: *const u8, ll: usize) {
    unsafe {
        ptr::copy_nonoverlapping(l, o, 16);
        if ll > 16 {
            spread::<M>(o.wrapping_add(16), l.wrapping_add(16), o.wrapping_add(ll));
        }
    }
}

/// `PAT32[o][i] = i % o`: the byte picks that spread an `o`-byte period (o < 16) over 32 bytes.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
static PAT32: [[u8; 32]; 16] = {
    let mut t = [[0u8; 32]; 16];
    let mut o = 1;
    while o < 16 {
        let mut i = 0;
        while i < 32 {
            t[o][i] = (i % o) as u8;
            i += 1;
        }
        o += 1;
    }
    t
};

/// `STRIDE32[o] = 32 - 32 % o`: the largest multiple of the period `o` that fits in 32 bytes (no divide per match).
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
static STRIDE32: [u8; 16] = {
    let mut t = [32u8; 16];
    let mut o = 1;
    while o < 16 {
        t[o] = (32 - 32 % o) as u8;
        o += 1;
    }
    t
};

/// A match at `offset` < 16 from the period in one ymm register: the `offset` bytes before `dst` are
/// spread over 32 bytes by one vpshufb on a broadcast 16-byte load; the same register is stored every
/// `32 - 32 % offset` bytes (a multiple of the period, so its phase holds), no load in the loop.
///
/// # Safety
/// As `short_match`; stores may run up to 31 bytes past `dst + len` (the callers' output slack, as
/// `spread`).
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline(never)]
unsafe fn short_match_wide(dst: *mut u8, offset: usize, len: usize) {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm256_broadcastsi128_si256, _mm256_loadu_si256, _mm256_shuffle_epi8, _mm256_storeu_si256,
        _mm_loadu_si128,
    };
    unsafe {
        let p = _mm256_shuffle_epi8(
            _mm256_broadcastsi128_si256(_mm_loadu_si128(dst.sub(offset) as *const __m128i)),
            _mm256_loadu_si256(PAT32.as_ptr().add(offset & 15) as *const __m256i),
        );
        let stride = *STRIDE32.get_unchecked(offset & 15) as usize;
        let end = dst.wrapping_add(len);
        let mut d = dst;
        loop {
            _mm256_storeu_si256(d as *mut __m256i, p);
            d = d.wrapping_add(stride);
            if d >= end {
                break;
            }
        }
    }
}

trait Mover {
    unsafe fn short(dst: *mut u8, offset: usize, len: usize) {
        unsafe { short_match(dst, offset, len) }
    }
    unsafe fn copy32(dst: *mut u8, src: *const u8);
    /// `dict_match` through this mover's own leaf.
    unsafe fn dict(sp: &Span, o: *mut u8, offset: usize, ml: usize) -> DictOut;
}

struct Narrow;

impl Mover for Narrow {
    #[inline(always)]
    unsafe fn dict(sp: &Span, o: *mut u8, offset: usize, ml: usize) -> DictOut {
        unsafe { dict_plain(sp, o, offset, ml) }
    }

    #[inline(always)]
    unsafe fn copy32(dst: *mut u8, src: *const u8) {
        unsafe {
            let a = ptr::read_unaligned(src as *const u128);
            let b = ptr::read_unaligned(src.add(16) as *const u128);
            ptr::write_unaligned(dst as *mut u128, a);
            ptr::write_unaligned(dst.add(16) as *mut u128, b);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
struct Wide;

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
impl Mover for Wide {
    #[inline(always)]
    unsafe fn short(dst: *mut u8, offset: usize, len: usize) {
        unsafe {
            if len > 32 {
                short_match_wide(dst, offset, len)
            } else {
                short_match(dst, offset, len)
            }
        }
    }

    #[inline(always)]
    unsafe fn dict(sp: &Span, o: *mut u8, offset: usize, ml: usize) -> DictOut {
        unsafe { dict_wide(sp, o, offset, ml) }
    }

    #[inline(always)]
    unsafe fn copy32(dst: *mut u8, src: *const u8) {
        use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256};
        unsafe { _mm256_storeu_si256(dst as *mut __m256i, _mm256_loadu_si256(src as *const __m256i)) }
    }
}

#[inline(always)]
unsafe fn far_match<M: Mover>(dst: *mut u8, src: *const u8, len: usize) {
    unsafe {
        M::copy32(dst, src);
        if len > 32 {
            spread::<M>(dst.wrapping_add(32), src.wrapping_add(32), dst.wrapping_add(len));
        }
    }
}

/// Result of a callee of `execute` as one scalar on every target (wasm32 returns a `Result` through
/// memory): true on success, otherwise false with the error left in the block for `failed`.
#[inline(always)]
unsafe fn status(st: *mut SeqState, r: Result<(), Error>) -> bool {
    match r {
        Ok(()) => true,
        Err(e) => {
            unsafe { (*st).span.err = Some(e) };
            false
        }
    }
}

#[inline(always)]
unsafe fn failed(st: *mut SeqState) -> Result<(), Error> {
    Err(unsafe { (*st).span.err }.unwrap_or(Error::Sequences))
}

/// Executes the sequence in `span.seq` at the span's literal and output positions.
#[inline(always)]
unsafe fn run(st: *mut SeqState) -> bool {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if unsafe { (*st).span.cpu.avx2 } {
        #[target_feature(enable = "avx2")]
        #[inline(never)]
        unsafe fn wide(st: *mut SeqState) -> bool {
            unsafe { status(st, run_impl::<Wide>(st)) }
        }
        return unsafe { wide(st) };
    }
    unsafe { run_plain(st) }
}

#[inline(never)]
unsafe fn run_plain(st: *mut SeqState) -> bool {
    unsafe { status(st, run_impl::<Narrow>(st)) }
}

#[inline(always)]
unsafe fn run_impl<M: Mover>(st: *mut SeqState) -> Result<(), Error> {
    unsafe {
        let sp = &mut (*st).span;
        let [ll, ml, offset] = sp.seq;
        let (ll, ml, offset) = (ll as usize, ml as usize, offset as usize);
        let (o, l) = (sp.dst as *mut u8, sp.lit as *const u8);
        if sp.lit_end - (l as usize) < ll || sp.out_end - (o as usize) < ll + ml {
            return Err(Error::Sequences);
        }
        literals::<M>(o, l, ll);
        sp.lit = l as usize + ll;
        let o = o.add(ll);
        if offset.wrapping_sub(1) < (o as usize - sp.start).min(sp.window) {
            let s = o.sub(offset);
            if offset >= 32 {
                far_match::<M>(o, s, ml);
            } else if offset >= 16 {
                wild16(o, s, ml);
            } else {
                M::short(o, offset, ml);
            }
        } else {
            dict_result(M::dict(sp, o, offset, ml))?;
        }
        sp.dst = o as usize + ml;
        Ok(())
    }
}

#[inline(always)]
unsafe fn short_match(dst: *mut u8, offset: usize, len: usize) {
    unsafe {
        let src = dst.sub(offset);
        if offset >= 8 {
            ptr::copy_nonoverlapping(src, dst, 8);
        } else {
            for i in 0..8 {
                *dst.add(i) = *src.add(i);
            }
        }
        if len > 8 {
            let dist = DIST[offset & 15];
            let end = dst.wrapping_add(len);
            let mut d = dst.wrapping_add(8);
            loop {
                ptr::copy_nonoverlapping(d.wrapping_sub(dist), d, 8);
                d = d.wrapping_add(8);
                if d >= end {
                    break;
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn recall(reps: *mut [u32; 4], ov: u32, ll: u32) -> u32 {
    unsafe {
        let fresh = ov > 3;
        let idx = select_unpredictable(fresh, 3, (ov.wrapping_sub(1) + (ll == 0) as u32) & 3) as usize;
        let offset = select_unpredictable(fresh, ov.wrapping_sub(3), *(*reps).get_unchecked(idx));
        let [r0, r1, r2, _] = *reps;
        let n1 = select_unpredictable(idx >= 1, r0, r1);
        let n2 = select_unpredictable(idx >= 2, r1, r2);
        *reps = [offset, n1, n2, offset.wrapping_sub(1)];
        offset
    }
}

/// Work block of `execute`: the stream, literal and output positions, block limits, decoder states
/// and repeat offsets, so the driver holds nothing across its calls and each callee takes the state
/// pointer alone.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct Span {
    /// Where the stream is read from: the stream, or the edge copy.
    src: usize,
    lit_end: usize,
    out_end: usize,
    /// Output address of the frame start.
    start: usize,
    window: usize,
    lit: usize,
    dst: usize,
    reps: [u32; 4],
    /// Bit position in `src` and the three FSE states.
    bp: usize,
    ls: usize,
    ms: usize,
    os: usize,
    /// Sequences left, the last one included.
    left: usize,
    /// Bit position the stream ends at: 0, or 64 once it is read from the edge copy.
    end: usize,
    /// A sequence (literal length, match length, offset) handed to `run`.
    seq: [u32; 3],
    dict: usize,
    dict_len: usize,
    ring: bool,
    cpu: Cpu,
    /// The error of a callee that returned false.
    err: Option<Error>,
}

const DONE: u32 = 0;
const WIDE: u32 = 1;
const FAR: u32 = 2;
const BAD: u32 = 3;

const ONE: i64 = 1 << 32;

/// Bits a sequence of the fused loop takes at most (wider ones leave it as `WIDE`). The loop starts an
/// iteration only with at least this many bits left: then its 64-bit read lies inside the stream
/// (byte index >= 0) and holds every bit the sequence takes.
const LIM: u32 = 57;

/// The main loop runs while at least this many bits are left and checks nothing else: the last
/// sequence reads only its extra bits (at most 31 + 16 + 16), so on a valid stream it stops before the
/// last sequence. A stream with more bits than its sequences take runs past the count and is rejected.
const LAST: u32 = 64;

/// Decodes and executes sequences, all but the last of `span.left`, until they are done (`EDGE`, in the
/// edge copy) or fewer than `LAST` bits are left (main loop). A call that ran into the last sequence
/// (the stream holds more bits than its sequences take) returns `BAD`.
#[inline(always)]
unsafe fn fused_impl<M: Mover, const EDGE: bool>(st: *mut SeqState) -> u32 {
    unsafe {
        if EDGE {
            if (*st).span.end == 0 {
                // The last bits: from here on the stream is read from its first 8 bytes copied above 8
                // zero bytes, so the loop's 64-bit reads stay inside the copy. Done in this leaf, so its
                // constants are not held across the driver's calls.
                // SAFETY: `edge` is 16 bytes owned by `st`; a stream not opened at the edge holds >= 8 bytes.
                let edge = (&raw mut (*st).edge).cast::<u8>();
                ptr::write_unaligned(edge.cast::<u64>(), 0);
                ptr::copy_nonoverlapping((*st).span.src as *const u8, edge.add(8), 8);
                ((*st).span.src, (*st).span.end) = (edge as usize, 64);
                (*st).span.bp += 64;
            }
            if (*st).span.bp < LIM as usize {
                // More bits were read than the stream holds.
                return BAD;
            }
        }
        let safe = (*st).span.left - 1;
        let mut bq = ((*st).span.bp as u64 | (safe as u64).wrapping_neg() << 32) as i64;
        let (mut lstate, mut mstate, mut ostate) = ((*st).span.ls, (*st).span.ms, (*st).span.os);
        let code = loop {
            let bp = bq as u32 as usize;
            let q = (bp + 7) >> 3;
            let sp = (*st).span.src as *const u8;
            let x = u64::from_le(ptr::read_unaligned(sp.add(q - 8) as *const u64)) << ((8 * q - bp) & 63);
            let ll_t = &(*st).ll;
            let ml_t = &(*st).ml;
            let of_t = &(*st).of;
            let (lv, ls) = (ll_t.val.get_unchecked(lstate), ll_t.step.get_unchecked(lstate));
            let (mv, ms) = (ml_t.val.get_unchecked(mstate), ml_t.step.get_unchecked(mstate));
            let (ov_, os) = (of_t.val.get_unchecked(ostate), of_t.step.get_unchecked(ostate));
            let a1 = os.nextra;
            let a2 = a1.wrapping_add(ms.nextra);
            let a3 = a2.wrapping_add(ls.nextra);
            let a4 = a3.wrapping_add(ls.nnb);
            let a6 = a4.wrapping_add(ms.nnb).wrapping_add(os.nnb);
            if a6 < -(LIM as i8) {
                break WIDE;
            }
            let ov = ov_.base | lane(x, a1, ov_.emask);
            let ml = (mv.base + lane(x, a2, mv.emask)) as usize;
            let ll = (lv.base + lane(x, a3, lv.emask)) as usize;
            lstate = (ls.next as u32 + lane(x, a4, ls.nmask)) as usize;
            mstate = (ms.next as u32 + lane(x, a4.wrapping_add(ptr::read(&ms.nnb)), ms.nmask)) as usize;
            ostate = (os.next as u32 + lane(x, a6, os.nmask)) as usize;
            bq += a6 as i64 + ONE;
            let offset = recall(&raw mut (*st).span.reps, ov, ll as u32) as usize;
            let span = &(*st).span;
            let o = span.dst as *mut u8;
            if offset.wrapping_sub(1) >= (o as usize + ll - span.start).min(span.window) {
                (*st).span.seq = [ll as u32, ml as u32, offset as u32];
                break FAR;
            }
            let l = span.lit as *const u8;
            if span.lit_end - (l as usize) < ll || span.out_end - (o as usize) < ll + ml {
                break BAD;
            }
            literals::<M>(o, l, ll);
            (*st).span.lit = l as usize + ll;
            let o = o.add(ll);
            (*st).span.dst = o as usize + ml;
            let m = o.sub(offset);
            if offset >= 32 {
                far_match::<M>(o, m, ml);
            } else if offset >= 16 {
                wild16(o, m, ml);
            } else {
                M::short(o, offset, ml);
            }
            if if EDGE { bq >= 0 || (bq as u32) < LIM } else { (bq as u32) < LAST } {
                break DONE;
            }
        };
        // Sequences of `safe` not done; a sequence that left the loop as `FAR` is decoded, not done.
        let counted = ((bq as u64 >> 32) as u32).wrapping_neg() as usize;
        let rem = if code == FAR { counted + 1 } else { counted };
        // `safe` again from the block (the loop does not write `left`): held, it waited in a stack slot.
        let safe = ptr::read(&raw const (*st).span.left) - 1;
        let sp = &mut (*st).span;
        sp.bp = bq as u32 as usize;
        (sp.ls, sp.ms, sp.os) = (lstate, mstate, ostate);
        sp.left = rem.wrapping_add(1);
        if rem > safe || (code != DONE && rem == 0) {
            BAD
        } else {
            code
        }
    }
}

#[inline(always)]
unsafe fn fused(st: *mut SeqState) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        let cpu = unsafe { (*st).span.cpu };
        if cpu.bmi2 && cpu.avx2 {
            #[target_feature(enable = "bmi1,bmi2,avx2")]
            #[inline(never)]
            unsafe fn wide(st: *mut SeqState) -> u32 {
                unsafe { fused_impl::<Wide, false>(st) }
            }
            return unsafe { wide(st) };
        }
        if cpu.bmi2 {
            #[target_feature(enable = "bmi1,bmi2")]
            #[inline(never)]
            unsafe fn narrow(st: *mut SeqState) -> u32 {
                unsafe { fused_impl::<Narrow, false>(st) }
            }
            return unsafe { narrow(st) };
        }
    }
    unsafe { fused_plain(st) }
}

#[inline(never)]
unsafe fn fused_plain(st: *mut SeqState) -> u32 {
    unsafe { fused_impl::<Narrow, false>(st) }
}

/// The sequences of the last bits, read from the edge copy (a few per block).
#[inline(never)]
unsafe fn fused_edge(st: *mut SeqState) -> u32 {
    unsafe { fused_impl::<Narrow, true>(st) }
}

/// Reads `n` bits below bit position `bp` of `s`, failing if fewer are left.
/// Safety: `s` holds at least 8 bytes and `bp <= 8 * len` (true for the stream and the edge copy, since
/// `open` sets `bp` below `8 * len` and reads only lower it), so the 8 bytes read lie inside `s`.
#[inline(always)]
unsafe fn take_careful(s: *const u8, bp: &mut usize, n: u32) -> Result<u64, Error> {
    if n as usize > *bp {
        return Err(Error::Sequences);
    }
    let q = (*bp + 7) >> 3;
    let p = q.saturating_sub(8);
    // SAFETY: p + 8 <= max(q, 8) <= len (see above).
    let w = u64::from_le(unsafe { ptr::read_unaligned(s.add(p).cast::<u64>()) });
    let v = peek(w, (64 + 8 * p - *bp) as u32, n);
    *bp -= n as usize;
    Ok(v)
}

/// Decodes one sequence with every read checked against the stream start, into `span.seq`; `UPDATE`
/// also reads the next states (every sequence but the last). A leaf: each value goes to the block as it
/// is read and each table entry is read where used, so only the state pointer and the bit position
/// stay in registers (read once up front, the entries and values waited in callee-saved registers).
#[inline(never)]
unsafe fn decode_careful<const UPDATE: bool>(st: *mut SeqState) -> bool {
    if unsafe { careful::<UPDATE>(st) }.is_some() {
        return true;
    }
    unsafe { (*st).span.err = Some(Error::Sequences) };
    false
}

/// Reads `n` bits below bit position `bp` of the block's stream, `None` if fewer are left.
/// Safety: as `take_careful`, for `span.src`.
#[inline(always)]
unsafe fn take_at(st: *mut SeqState, bp: &mut usize, n: u32) -> Option<u64> {
    if n as usize > *bp {
        return None;
    }
    // SAFETY: `span.src` is the stream or the edge copy, both live during `execute`.
    let s = at(unsafe { &(*st).span.src }) as *const u8;
    let q = (*bp + 7) >> 3;
    let p = q.saturating_sub(8);
    // SAFETY: p + 8 <= max(q, 8) <= len (see `take_careful`).
    let w = u64::from_le(unsafe { ptr::read_unaligned(s.add(p).cast::<u64>()) });
    let v = peek(w, (64 + 8 * p - *bp) as u32, n);
    *bp -= n as usize;
    Some(v)
}

#[inline(always)]
unsafe fn careful<const UPDATE: bool>(st: *mut SeqState) -> Option<()> {
    unsafe {
        let mut bp = (*st).span.bp;
        // Indices and entries read where used (volatile, not hoisted to the top).
        let li = move || at(&(*st).span.ls) & 511;
        let mi = move || at(&(*st).span.ms) & 511;
        let oi = move || at(&(*st).span.os) & 255;
        let n = at((*st).of.step.get_unchecked(oi())).extra() + at((*st).ml.step.get_unchecked(mi())).extra();
        let v = take_at(st, &mut bp, n)?;
        let mlb = at((*st).ml.step.get_unchecked(mi())).extra();
        put(&mut (*st).span.seq[2], at(&(*st).of.val.get_unchecked(oi()).base).wrapping_add((v >> mlb) as u32));
        put(&mut (*st).span.seq[1], at(&(*st).ml.val.get_unchecked(mi()).base).wrapping_add(low(v, mlb) as u32));
        let v = take_at(st, &mut bp, at((*st).ll.step.get_unchecked(li())).extra())?;
        put(&mut (*st).span.seq[0], at(&(*st).ll.val.get_unchecked(li()).base).wrapping_add(v as u32));
        if UPDATE {
            // Each new state is stored once its old one is no longer read.
            let v = take_at(st, &mut bp, at((*st).ll.step.get_unchecked(li())).nb())?;
            put(&mut (*st).span.ls, at((*st).ll.step.get_unchecked(li())).next() + v as usize);
            let v = take_at(st, &mut bp, at((*st).ml.step.get_unchecked(mi())).nb())?;
            put(&mut (*st).span.ms, at((*st).ml.step.get_unchecked(mi())).next() + v as usize);
            let v = take_at(st, &mut bp, at((*st).of.step.get_unchecked(oi())).nb())?;
            put(&mut (*st).span.os, at((*st).of.step.get_unchecked(oi())).next() + v as usize);
        }
        (*st).span.bp = bp;
        // Read back from the block, so they do not wait in registers through the state reads.
        let (ll, ov) = (at(&(*st).span.seq[0]), at(&(*st).span.seq[2]));
        (*st).span.seq[2] = recall(&raw mut (*st).span.reps, ov, ll);
        Some(())
    }
}

/// Binds the block of job `j` and the sequence count `parse_header` read into the work block: limits,
/// literal and output positions, repeat offsets; `execute` then runs it and `out` gives the output
/// position. Each field goes from the job to the span on its own, so the caller holds none of them.
/// Safety: the caller checked `fs <= pos` and `pos + block_max + SLACK <= cap` (the output limit and the
/// copies' slack), and `lit_len + SLACK` bytes are readable at `lit`.
#[inline(always)]
pub(crate) unsafe fn bind(st: &mut SeqState, j: &Job, cpu: Cpu) {
    // One field at a time, each store where its value is computed (stores of one struct value were
    // issued after all its loads, more values than the volatile registers).
    put(&mut st.span.lit, j.lit);
    put(&mut st.span.lit_end, j.lit + j.lit_len);
    put(&mut st.span.start, at(&j.base) + j.fs);
    let d = at(&j.base) + j.pos;
    put(&mut st.span.dst, d);
    put(&mut st.span.out_end, d + j.block_max);
    put(&mut st.span.window, j.window);
    put(&mut st.span.dict, j.ext);
    put(&mut st.span.dict_len, j.ext_len);
    put(&mut st.span.ring, j.ring);
    put(&mut st.span.cpu, cpu);
    put(&mut st.span.left, at(&st.head.nb));
    let [r0, r1, r2] = st.reps;
    let sp = &mut st.span;
    sp.reps = [r0, r1, r2, r0.wrapping_sub(1)];
    (sp.src, sp.bp, sp.ls, sp.ms, sp.os, sp.end, sp.seq, sp.err) = (0, 0, 0, 0, 0, 0, [0; 3], None);
}

impl SeqState {
    /// Output position after `execute`, relative to `base`.
    pub(crate) fn out(&self, base: *mut u8) -> usize {
        self.span.dst - base as usize
    }
}

/// Decodes and executes the sequences of the block bound by `bind` from `src`, then copies the
/// literals left. Every value lives in the work block, so nothing but the state pointer is held
/// across the calls.
#[inline(never)]
pub(crate) unsafe fn execute(st: &mut SeqState, src: &[u8]) -> Result<(), Error> {
    let st: *mut SeqState = st;
    unsafe {
        if !open(st, src) {
            return failed(st);
        }
        // Every sequence but the last (it reads no state update) runs in the fused loop.
        while (*st).span.left > 1 {
            // The main loop while at least `LAST` bits are left in the stream; the edge loop after.
            let code = if (*st).span.end == 0 && (*st).span.bp >= LAST as usize { fused(st) } else { fused_edge(st) };
            // A range test and a bit test, not a switch: its jump table address would be held across the
            // calls.
            if code != DONE {
                if code > FAR {
                    return Err(Error::Sequences);
                }
                if (code & WIDE != 0 && !decode_careful::<true>(st)) || !run(st) {
                    return failed(st);
                }
                (*st).span.left -= 1;
            }
        }
        if (*st).span.left == 1 && (!decode_careful::<false>(st) || !run(st)) {
            return failed(st);
        }
        let sp = &mut (*st).span;
        if sp.bp != sp.end {
            return Err(Error::Sequences);
        }
        let rest = sp.lit_end - sp.lit;
        if rest > sp.out_end - sp.dst {
            return Err(Error::Sequences);
        }
        // Stored before the copy, so nothing is held across it.
        let (l, d) = (sp.lit as *const u8, sp.dst as *mut u8);
        sp.dst += rest;
        let [r0, r1, r2, _] = sp.reps;
        (*st).reps = [r0, r1, r2];
        ptr::copy_nonoverlapping(l, d, rest);
        Ok(())
    }
}

/// Opens the stream: finds its end marker, points the block at it (a stream under 8 bytes at the edge
/// copy) and reads the three initial states. A leaf, so its temporaries use the volatile registers.
#[inline(never)]
unsafe fn open(st: *mut SeqState, src: &[u8]) -> bool {
    unsafe { status(st, opened(st, src)) }
}

#[inline(always)]
unsafe fn opened(st: *mut SeqState, src: &[u8]) -> Result<(), Error> {
    let len = src.len();
    if len >= 1 << 28 {
        return Err(Error::Sequences);
    }
    let top = match src.last() {
        Some(&t) if t != 0 => t,
        _ => return Err(Error::Bitstream),
    };
    unsafe {
        let edge = (&raw mut (*st).edge).cast::<u8>();
        // A stream shorter than 8 bytes is read from the edge copy from the start.
        let short = len < 8;
        // The end marker is the highest set bit of the last byte.
        let mut bp = 8 * len - (top.leading_zeros() as usize + 1);
        if short {
            // From the last byte down, so no shift count is kept: byte i ends at bits 8 * i.
            let mut w = 0u64;
            for &b in src.iter().rev() {
                w = w << 8 | b as u64;
            }
            ptr::write_unaligned(edge.cast::<u64>(), 0);
            ptr::write_unaligned(edge.add(8).cast::<u64>(), w.to_le());
            // The same position from the gathered word (its highest set bit is the marker), 64 bits up,
            // so the last byte is not held across the loop.
            bp = 127 - w.leading_zeros() as usize;
        }
        // Selected values, one store each: stores in both branches are merged through field offsets
        // held in registers.
        let s = if short { edge as *const u8 } else { src.as_ptr() };
        (*st).span.src = s as usize;
        (*st).span.end = if short { 64 } else { 0 };
        (*st).span.ls = take_careful(s, &mut bp, (*st).ll.log)? as usize;
        (*st).span.os = take_careful(s, &mut bp, (*st).of.log)? as usize;
        (*st).span.ms = take_careful(s, &mut bp, (*st).ml.log)? as usize;
        (*st).span.bp = bp;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_free_every_length() {
        let src: alloc::vec::Vec<u8> = (0..160u8).map(|i| i.wrapping_mul(37) ^ 0x5A).collect();
        for n in 0..=130 {
            for (s, d) in [(0, 0), (1, 3), (7, 5)] {
                let mut dst = [0xEEu8; 160];
                // SAFETY: `s + n` and `d + n` are within both 160-byte buffers.
                unsafe { copy_free::<Narrow>(src.as_ptr().add(s), dst.as_mut_ptr().add(d), n) };
                assert_eq!(&dst[d..d + n], &src[s..s + n], "n {n}");
                assert!(dst[..d].iter().chain(&dst[d + n..]).all(|&b| b == 0xEE), "n {n} wrote outside");
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                if Cpu::detect().avx2 {
                    #[target_feature(enable = "avx2")]
                    unsafe fn wide(src: *const u8, dst: *mut u8, n: usize) {
                        unsafe { copy_free::<Wide>(src, dst, n) }
                    }
                    let mut dst = [0xEEu8; 160];
                    // SAFETY: as above; the CPU has AVX2.
                    unsafe { wide(src.as_ptr().add(s), dst.as_mut_ptr().add(d), n) };
                    assert_eq!(&dst[d..d + n], &src[s..s + n], "wide n {n}");
                    assert!(dst[..d].iter().chain(&dst[d + n..]).all(|&b| b == 0xEE), "wide n {n} wrote outside");
                }
            }
        }
    }
}
