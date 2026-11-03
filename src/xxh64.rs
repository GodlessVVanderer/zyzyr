const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

#[inline(always)]
fn round(acc: u64, lane: u64) -> u64 {
    acc.wrapping_add(lane.wrapping_mul(P2))
        .rotate_left(31)
        .wrapping_mul(P1)
}

#[inline(always)]
fn merge(acc: u64, v: u64) -> u64 {
    (acc ^ round(0, v)).wrapping_mul(P1).wrapping_add(P4)
}

#[inline(always)]
fn lane64(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3], b[i + 4], b[i + 5], b[i + 6], b[i + 7]])
}

#[inline(always)]
fn lane32(b: &[u8], i: usize) -> u64 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as u64
}

/// Copies `n <= 32` bytes from `src` to `dst` inline, as two loads and two stores of the widest width that
/// fits, which may overlap in the middle.
///
/// # Safety
/// As `ptr::copy_nonoverlapping(src, dst, n)`, with `n <= 32`.
#[inline(always)]
unsafe fn short(src: *const u8, dst: *mut u8, n: usize) {
    // SAFETY: by the caller; every access lies in the first `n` bytes of `src` or `dst`.
    unsafe {
        if n >= 16 {
            let (a, b) = (src.cast::<u128>().read_unaligned(), src.add(n - 16).cast::<u128>().read_unaligned());
            dst.cast::<u128>().write_unaligned(a);
            dst.add(n - 16).cast::<u128>().write_unaligned(b);
        } else if n >= 8 {
            let (a, b) = (src.cast::<u64>().read_unaligned(), src.add(n - 8).cast::<u64>().read_unaligned());
            dst.cast::<u64>().write_unaligned(a);
            dst.add(n - 8).cast::<u64>().write_unaligned(b);
        } else if n >= 4 {
            let (a, b) = (src.cast::<u32>().read_unaligned(), src.add(n - 4).cast::<u32>().read_unaligned());
            dst.cast::<u32>().write_unaligned(a);
            dst.add(n - 4).cast::<u32>().write_unaligned(b);
        } else if n >= 2 {
            let (a, b) = (src.cast::<u16>().read_unaligned(), src.add(n - 2).cast::<u16>().read_unaligned());
            dst.cast::<u16>().write_unaligned(a);
            dst.add(n - 2).cast::<u16>().write_unaligned(b);
        } else if n == 1 {
            dst.write(src.read());
        }
    }
}

#[derive(Clone)]
pub struct Xxh64 {
    v: [u64; 4],
    buf: [u8; 32],
    buf_len: usize,
    total: u64,
    seed: u64,
}

impl Xxh64 {
    pub fn new(seed: u64) -> Self {
        Xxh64 {
            v: [
                seed.wrapping_add(P1).wrapping_add(P2),
                seed.wrapping_add(P2),
                seed,
                seed.wrapping_sub(P1),
            ],
            buf: [0; 32],
            buf_len: 0,
            total: 0,
            seed,
        }
    }

    #[inline(always)]
    fn stripes(v: &mut [u64; 4], data: &[u8]) {
        let [mut a, mut b, mut c, mut d] = *v;
        for s in data.as_chunks::<32>().0 {
            a = round(a, lane64(s, 0));
            b = round(b, lane64(s, 8));
            c = round(c, lane64(s, 16));
            d = round(d, lane64(s, 24));
        }
        *v = [a, b, c, d];
    }

    /// Hashes `data` after the bytes before it, buffering a partial stripe. Call-free: the buffer copies (under
    /// 32 bytes) are inline (`copy_from_slice` called `memcpy` for each), and the buffered stripe is hashed in
    /// place (copied out first, it went through the stack).
    pub fn update(&mut self, mut data: &[u8]) {
        self.total = self.total.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (32 - self.buf_len).min(data.len());
            // SAFETY: `buf_len + take <= 32` and `take <= data.len()`.
            unsafe { short(data.as_ptr(), self.buf.as_mut_ptr().add(self.buf_len), take) };
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len < 32 {
                return;
            }
            Self::stripes(&mut self.v, &self.buf);
            self.buf_len = 0;
        }
        let whole = data.len() & !31;
        Self::stripes(&mut self.v, &data[..whole]);
        let rest = &data[whole..];
        // SAFETY: `rest.len() < 32`.
        unsafe { short(rest.as_ptr(), self.buf.as_mut_ptr(), rest.len()) };
        self.buf_len = rest.len();
    }

    /// Hashes the whole 32-byte stripes of `data` (its first `data.len() & !31` bytes) where it lies, no
    /// buffer; the rest is handed again with the next data or to `finish_with`. Not mixed with `update`.
    /// The loop keeps both primes and its end in registers (9 values; 3 callee-saved registers on Win64):
    /// read from memory instead, so the loop fits the 7 volatile registers, the hash after each 128 KB
    /// block cost about 2.5% more, 0.3% of the decode.
    #[inline(always)]
    pub fn absorb(&mut self, data: &[u8]) {
        let whole = data.len() & !31;
        self.total = self.total.wrapping_add(whole as u64);
        Self::stripes(&mut self.v, &data[..whole]);
    }

    /// The hash of the absorbed stripes followed by `tail` (fewer than 32 bytes).
    pub fn finish_with(&self, tail: &[u8]) -> u64 {
        Self::end(self.v, self.seed, self.total.wrapping_add(tail.len() as u64), tail)
    }

    pub fn finish(&self) -> u64 {
        Self::end(self.v, self.seed, self.total, &self.buf[..self.buf_len])
    }

    fn end(v: [u64; 4], seed: u64, total: u64, tail: &[u8]) -> u64 {
        let mut h = if total >= 32 {
            let [a, b, c, d] = v;
            let mut h = a
                .rotate_left(1)
                .wrapping_add(b.rotate_left(7))
                .wrapping_add(c.rotate_left(12))
                .wrapping_add(d.rotate_left(18));
            h = merge(h, a);
            h = merge(h, b);
            h = merge(h, c);
            merge(h, d)
        } else {
            seed.wrapping_add(P5)
        };
        h = h.wrapping_add(total);
        let mut i = 0;
        while i + 8 <= tail.len() {
            h ^= round(0, lane64(tail, i));
            h = h.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
            i += 8;
        }
        if i + 4 <= tail.len() {
            h ^= lane32(tail, i).wrapping_mul(P1);
            h = h.rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
            i += 4;
        }
        while i < tail.len() {
            h ^= (tail[i] as u64).wrapping_mul(P5);
            h = h.rotate_left(11).wrapping_mul(P1);
            i += 1;
        }
        h ^= h >> 33;
        h = h.wrapping_mul(P2);
        h ^= h >> 29;
        h = h.wrapping_mul(P3);
        h ^= h >> 32;
        h
    }
}

#[cfg(test)]
pub fn xxh64(data: &[u8], seed: u64) -> u64 {
    let mut s = Xxh64::new(seed);
    s.update(data);
    s.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn ramp(n: usize) -> Vec<u8> {
        (0..n).map(|i| ((i * 31 + 7) % 256) as u8).collect()
    }

    #[test]
    fn full_values() {
        assert_eq!(xxh64(b"", 0), 0xEF46_DB37_51D8_E999);
        assert_eq!(xxh64(b"abc", 0), 0x44BC_2CF5_AD77_0999);
    }

    #[test]
    fn low_halves_from_reference_frames() {
        let known: [(usize, u32); 14] = [
            (0, 0x51D8_E999),
            (1, 0xE858_BBB7),
            (3, 0x32A4_87F9),
            (4, 0xE3FF_8F04),
            (7, 0x6C6F_9A8E),
            (8, 0x2696_83E0),
            (15, 0x9357_CAA7),
            (31, 0xA39A_D4A1),
            (32, 0x671C_C43D),
            (33, 0xED85_7664),
            (63, 0x2707_057F),
            (64, 0x5729_D17E),
            (100, 0x3E70_C151),
            (1000, 0x2804_3D35),
        ];
        for (n, want) in known {
            assert_eq!(xxh64(&ramp(n), 0) as u32, want, "length {n}");
        }
    }

    #[test]
    fn streaming_matches_one_shot() {
        let data = ramp(5000);
        let whole = xxh64(&data, 0);
        for step in [1usize, 3, 7, 31, 32, 33, 64, 100, 999] {
            let mut s = Xxh64::new(0);
            for c in data.chunks(step) {
                s.update(c);
            }
            assert_eq!(s.finish(), whole, "chunk {step}");
        }
    }

    #[test]
    fn in_place_matches_one_shot() {
        let data = ramp(5000);
        for n in [0usize, 1, 31, 32, 33, 1000, 5000] {
            for step in [1usize, 7, 32, 100, 1024] {
                let mut s = Xxh64::new(0);
                let (mut hashed, mut end) = (0, 0);
                while end < n {
                    end = (end + step).min(n);
                    s.absorb(&data[hashed..end]);
                    hashed += (end - hashed) & !31;
                }
                assert_eq!(s.finish_with(&data[hashed..n]), xxh64(&data[..n], 0), "length {n} chunk {step}");
            }
        }
    }
}
