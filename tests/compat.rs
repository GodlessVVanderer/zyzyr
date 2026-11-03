mod common;

use std::ffi::{c_char, c_int, c_uint, c_ulonglong, c_void, CStr};
use std::ptr;

use zyzyr::compat::{ZstdInBuffer, ZstdOutBuffer};

#[repr(C)]
struct ZstdDCtx {
    _opaque: [u8; 0],
}

extern "C" {
    fn ZSTD_decompress(dst: *mut c_void, dst_capacity: usize, src: *const c_void, compressed_size: usize) -> usize;
    fn ZSTD_getFrameContentSize(src: *const c_void, src_size: usize) -> c_ulonglong;
    fn ZSTD_findFrameCompressedSize(src: *const c_void, src_size: usize) -> usize;
    fn ZSTD_isError(code: usize) -> c_uint;
    fn ZSTD_getErrorName(code: usize) -> *const c_char;
    fn ZSTD_createDCtx() -> *mut ZstdDCtx;
    fn ZSTD_freeDCtx(dctx: *mut ZstdDCtx) -> usize;
    fn ZSTD_decompressDCtx(dctx: *mut ZstdDCtx, dst: *mut c_void, cap: usize, src: *const c_void, size: usize) -> usize;
    fn ZSTD_decompress_usingDict(
        dctx: *mut ZstdDCtx,
        dst: *mut c_void,
        cap: usize,
        src: *const c_void,
        size: usize,
        dict: *const c_void,
        dict_size: usize,
    ) -> usize;
    fn ZSTD_DCtx_loadDictionary(dctx: *mut ZstdDCtx, dict: *const c_void, dict_size: usize) -> usize;
    fn ZSTD_DCtx_setParameter(dctx: *mut ZstdDCtx, param: c_int, value: c_int) -> usize;
    fn ZSTD_DCtx_reset(dctx: *mut ZstdDCtx, reset: c_int) -> usize;
    fn ZSTD_createDStream() -> *mut ZstdDCtx;
    fn ZSTD_freeDStream(zds: *mut ZstdDCtx) -> usize;
    fn ZSTD_initDStream(zds: *mut ZstdDCtx) -> usize;
    fn ZSTD_decompressStream(zds: *mut ZstdDCtx, output: *mut ZstdOutBuffer, input: *mut ZstdInBuffer) -> usize;
    fn ZSTD_DStreamInSize() -> usize;
    fn ZSTD_DStreamOutSize() -> usize;
}

const MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

fn raw_frame(payload: &[u8], with_size: bool) -> Vec<u8> {
    let mut v = MAGIC.to_vec();
    if with_size {
        v.extend_from_slice(&[0x20, payload.len() as u8]);
    } else {
        v.extend_from_slice(&[0x00, 0x00]);
    }
    let bh = ((payload.len() << 3) | 1) as u32;
    v.extend_from_slice(&bh.to_le_bytes()[..3]);
    v.extend_from_slice(payload);
    v
}

fn name(code: usize) -> String {
    unsafe { CStr::from_ptr(ZSTD_getErrorName(code)) }.to_string_lossy().into_owned()
}

fn error_code(r: usize) -> usize {
    assert_eq!(unsafe { ZSTD_isError(r) }, 1, "expected an error, got {r}");
    r.wrapping_neg()
}

fn pairs() -> Vec<(Vec<u8>, Vec<u8>, String)> {
    let mut v = Vec::new();
    for c in common::cases().into_iter().filter(|c| c.dict.is_none()) {
        let want = common::decode(&c).unwrap();
        assert!(c.check(&want), "{}", c.name);
        v.push((c.data, want, c.name));
    }
    v
}

#[test]
fn one_shot_and_queries() {
    let f = raw_frame(b"hello compat", true);
    let mut out = [0u8; 32];
    unsafe {
        let r = ZSTD_decompress(out.as_mut_ptr().cast(), out.len(), f.as_ptr().cast(), f.len());
        assert_eq!(r, 12);
        assert_eq!(&out[..12], b"hello compat");
        let r = ZSTD_decompress(out.as_mut_ptr().cast(), 5, f.as_ptr().cast(), f.len());
        assert_eq!(error_code(r), 70);
        assert_eq!(name(r), "Destination buffer is too small");
        let r = ZSTD_decompress(out.as_mut_ptr().cast(), 32, b"nope".as_ptr().cast(), 4);
        assert_eq!(error_code(r), 10);
        let r = ZSTD_decompress(ptr::null_mut(), 32, f.as_ptr().cast(), f.len());
        assert_eq!(error_code(r), 74);
        let r = ZSTD_decompress(out.as_mut_ptr().cast(), 32, f.as_ptr().cast(), f.len() - 2);
        assert_eq!(error_code(r), 72);
        assert_eq!(ZSTD_isError(0), 0);
        assert_eq!(name(12), "No error detected");
        assert_eq!(ZSTD_getFrameContentSize(f.as_ptr().cast(), f.len()), 12);
        let open = raw_frame(b"abc", false);
        assert_eq!(ZSTD_getFrameContentSize(open.as_ptr().cast(), open.len()), !0);
        assert_eq!(ZSTD_getFrameContentSize(b"xxxxxxxx".as_ptr().cast(), 8), !0 - 1);
        let mut two = f.clone();
        two.extend_from_slice(&open);
        assert_eq!(ZSTD_findFrameCompressedSize(two.as_ptr().cast(), two.len()), f.len());
        assert_eq!(error_code(ZSTD_findFrameCompressedSize(two.as_ptr().cast(), 7)), 72);
        assert!(ZSTD_DStreamInSize() > 0 && ZSTD_DStreamOutSize() > 0);
    }
}

#[test]
fn dctx_parameters_and_files() {
    unsafe {
        let d = ZSTD_createDCtx();
        assert!(!d.is_null());
        assert_eq!(ZSTD_DCtx_setParameter(d, 100, 31), 0);
        assert_eq!(error_code(ZSTD_DCtx_setParameter(d, 1000, 1)), 40);
        assert_eq!(ZSTD_DCtx_reset(d, 3), 0);
        assert_eq!(error_code(ZSTD_DCtx_reset(d, 9)), 40);
        for (z, want, n) in pairs() {
            let mut out = vec![0u8; want.len() + 1];
            let r = ZSTD_decompressDCtx(d, out.as_mut_ptr().cast(), out.len(), z.as_ptr().cast(), z.len());
            assert_eq!(ZSTD_isError(r), 0, "{n}: {}", name(r));
            assert_eq!(r, want.len(), "{n}");
            assert!(out[..r] == want[..], "{n}");
            println!("{n}: {} bytes through ZSTD_decompressDCtx", r);
        }
        assert_eq!(ZSTD_freeDCtx(d), 0);
        assert_eq!(ZSTD_freeDCtx(ptr::null_mut()), 0);
    }
}

fn stream_all(d: *mut ZstdDCtx, z: &[u8], in_chunk: usize, out_cap: usize) -> (Vec<u8>, usize) {
    let mut got = Vec::new();
    let mut buf = vec![0u8; out_cap];
    let mut last = 0usize;
    unsafe {
        assert_eq!(ZSTD_isError(ZSTD_initDStream(d)), 0);
        for chunk in z.chunks(in_chunk) {
            let mut input = ZstdInBuffer { src: chunk.as_ptr().cast(), size: chunk.len(), pos: 0 };
            while input.pos < input.size {
                let mut output = ZstdOutBuffer { dst: buf.as_mut_ptr().cast(), size: buf.len(), pos: 0 };
                let r = ZSTD_decompressStream(d, &mut output, &mut input);
                assert_eq!(ZSTD_isError(r), 0, "{}", name(r));
                got.extend_from_slice(&buf[..output.pos]);
                last = r;
            }
        }
        while last != 0 {
            let mut input = ZstdInBuffer { src: ptr::null(), size: 0, pos: 0 };
            let mut output = ZstdOutBuffer { dst: buf.as_mut_ptr().cast(), size: buf.len(), pos: 0 };
            let r = ZSTD_decompressStream(d, &mut output, &mut input);
            assert_eq!(ZSTD_isError(r), 0, "{}", name(r));
            got.extend_from_slice(&buf[..output.pos]);
            if output.pos == 0 && r == last {
                break;
            }
            last = r;
        }
    }
    (got, last)
}

#[test]
fn streaming_matches_one_shot() {
    let mut frames = raw_frame(b"first", true);
    frames.extend_from_slice(&[0x50, 0x2A, 0x4D, 0x18, 3, 0, 0, 0, 7, 7, 7]);
    frames.extend_from_slice(&raw_frame(b"second frame", false));
    let mut cases: Vec<(Vec<u8>, Vec<u8>, String)> = vec![(frames, b"firstsecond frame".to_vec(), "built".into())];
    cases.extend(pairs());
    unsafe {
        let d = ZSTD_createDStream();
        for (z, want, n) in &cases {
            for (ic, oc) in [(1usize, 7usize), (13, 1), (4096, 100), (131_075, 131_072), (z.len().max(1), 64)] {
                if z.len() > 300_000 && (ic < 4096 || oc < 100) {
                    continue;
                }
                let (got, last) = stream_all(d, z, ic, oc);
                assert_eq!(last, 0, "{n} in {ic} out {oc}: stream did not end on a frame boundary");
                assert!(got == *want, "{n} in {ic} out {oc}: {} bytes vs {}", got.len(), want.len());
            }
            println!("{n}: streamed in five chunkings");
        }
        let (z, _, _) = &cases[0];
        let (_, last) = stream_all(d, &z[..z.len() - 3], 5, 9);
        assert_ne!(last, 0);
        let bad = b"definitely not a frame".to_vec();
        let mut input = ZstdInBuffer { src: bad.as_ptr().cast(), size: bad.len(), pos: 0 };
        let mut out = [0u8; 16];
        let mut output = ZstdOutBuffer { dst: out.as_mut_ptr().cast(), size: 16, pos: 0 };
        assert_eq!(ZSTD_initDStream(d), ZSTD_DStreamInSize());
        assert_eq!(error_code(ZSTD_decompressStream(d, &mut output, &mut input)), 10);
        assert_eq!(error_code(ZSTD_decompressStream(ptr::null_mut(), &mut output, &mut input)), 1);
        assert_eq!(ZSTD_freeDStream(d), 0);
    }
}

#[test]
fn dictionaries() {
    unsafe {
        let d = ZSTD_createDCtx();
        let mut n = 0;
        for c in common::cases() {
            let Some(dict) = &c.dict else {
                continue;
            };
            let z = &c.data;
            let mut out = vec![0u8; c.len];
            let r = ZSTD_decompressDCtx(d, out.as_mut_ptr().cast(), out.len(), z.as_ptr().cast(), z.len());
            assert_eq!(ZSTD_isError(r), 1, "{}", c.name);
            let r = ZSTD_decompress_usingDict(d, out.as_mut_ptr().cast(), out.len(), z.as_ptr().cast(), z.len(), dict.as_ptr().cast(), dict.len());
            assert_eq!(r, c.len, "{}: {}", c.name, name(r));
            assert!(c.check(&out));
            assert_eq!(ZSTD_DCtx_loadDictionary(d, dict.as_ptr().cast(), dict.len()), 0);
            let (got, last) = stream_all(d, z, 17, 333);
            assert_eq!(last, 0);
            assert!(c.check(&got));
            assert_eq!(ZSTD_DCtx_reset(d, 2), 0);
            n += 1;
        }
        ZSTD_freeDCtx(d);
        assert!(n >= 3);
        println!("{n} dictionary frames through the compat functions");
    }
}
