use crate::Error;

#[inline(always)]
pub(crate) fn le64(src: &[u8], pos: usize) -> u64 {
    let b: &[u8; 8] = src[pos..pos + 8].try_into().unwrap();
    u64::from_le_bytes(*b)
}

#[inline(always)]
pub(crate) fn peek(bits: u64, used: u32, n: u32) -> u64 {
    (bits.wrapping_shl(used) >> 1) >> (63 - n)
}

#[derive(Clone, Copy)]
pub(crate) struct Back<'a> {
    pub src: &'a [u8],
    pub pos: usize,
    pub bits: u64,
    pub used: u32,
}

impl<'a> Back<'a> {
    pub(crate) fn new(src: &'a [u8], pad: &'a mut [u8; 8]) -> Result<Self, Error> {
        let len = src.len();
        if len == 0 || src[len - 1] == 0 {
            return Err(Error::Bitstream);
        }
        let s: &'a [u8] = if len >= 8 {
            src
        } else {
            pad.fill(0);
            pad[..len].copy_from_slice(src);
            &pad[..]
        };
        let pos = s.len() - 8;
        let bits = le64(s, pos);
        Ok(Back {
            src: s,
            pos,
            bits,
            used: bits.leading_zeros() + 1,
        })
    }

    #[inline(always)]
    pub(crate) fn refill(&mut self) {
        let step = ((self.used >> 3) as usize).min(self.pos);
        self.pos -= step;
        self.used -= (step as u32) << 3;
        self.bits = le64(self.src, self.pos);
    }

    #[cfg(test)]
    pub(crate) fn read(&mut self, n: u32) -> u64 {
        let v = peek(self.bits, self.used, n);
        self.used += n;
        v
    }

    pub(crate) fn finish(&mut self) -> bool {
        self.refill();
        self.pos == 0 && self.used == 64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn spec_example_bits() {
        let data = [0b0001_0000u8, 0b0000_1101];
        let mut pad = [0u8; 8];
        let mut r = Back::new(&data, &mut pad).unwrap();
        assert_eq!(r.read(1), 1);
        assert_eq!(r.read(2), 0b01);
        assert_eq!(r.read(4), 0b0001);
        assert_eq!(r.read(4), 0b0000);
        assert!(r.finish());
    }

    #[test]
    fn long_stream_refills() {
        let mut data = vec![0xA5u8; 40];
        data[39] = 0x01;
        let mut pad = [0u8; 8];
        let mut r = Back::new(&data, &mut pad).unwrap();
        let mut total = 0u32;
        while total + 7 <= 39 * 8 {
            r.refill();
            assert!(r.read(7) < 128);
            total += 7;
        }
        r.refill();
        r.read(39 * 8 - total);
        assert!(r.finish());
        r.read(1);
        assert!(!r.finish());
    }

    #[test]
    fn zero_last_byte_rejected() {
        let mut pad = [0u8; 8];
        assert!(Back::new(&[1, 2, 3, 0, 0, 0, 0, 0], &mut pad).is_err());
        assert!(Back::new(&[], &mut pad).is_err());
        assert!(Back::new(&[7, 0], &mut pad).is_err());
    }
}
