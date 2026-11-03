mod common;

use std::ffi::CStr;
use std::ptr;

use zyzyr::capi::*;

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

fn code(s: ZyzyrStatus) -> i32 {
    s.code
}

#[test]
fn one_shot_exact_short_and_query() {
    let f = raw_frame(b"hello, world", true);
    let mut out = [0u8; 12];
    let mut n = usize::MAX;
    let s = unsafe { zyzyr_decompress(f.as_ptr(), f.len(), out.as_mut_ptr(), out.len(), &mut n) };
    assert_eq!(s, ZyzyrStatus::OK);
    assert_eq!((n, &out[..]), (12, &b"hello, world"[..]));

    let mut small = [0u8; 5];
    let s = unsafe { zyzyr_decompress(f.as_ptr(), f.len(), small.as_mut_ptr(), small.len(), &mut n) };
    assert_eq!(s, ZyzyrStatus::SHORT_BUFFER);
    assert_eq!(n, 12);
    assert_eq!(small, [0u8; 5]);

    let s = unsafe { zyzyr_decompress(f.as_ptr(), f.len(), ptr::null_mut(), 0, &mut n) };
    assert_eq!(s, ZyzyrStatus::SHORT_BUFFER);
    assert_eq!(n, 12);

    let empty = raw_frame(b"", true);
    let s = unsafe { zyzyr_decompress(empty.as_ptr(), empty.len(), ptr::null_mut(), 0, &mut n) };
    assert_eq!(s, ZyzyrStatus::OK);
    assert_eq!(n, 0);
}

#[test]
fn null_and_length_checks() {
    let f = raw_frame(b"abc", true);
    let mut out = [0u8; 8];
    let mut n = 7usize;
    unsafe {
        assert_eq!(zyzyr_decompress(ptr::null(), 3, out.as_mut_ptr(), 8, &mut n), ZyzyrStatus::NULL);
        assert_eq!(n, 0);
        assert_eq!(zyzyr_decompress(f.as_ptr(), f.len(), out.as_mut_ptr(), 8, ptr::null_mut()), ZyzyrStatus::NULL);
        assert_eq!(zyzyr_decompress(f.as_ptr(), f.len(), ptr::null_mut(), 8, &mut n), ZyzyrStatus::NULL);
        let dangling = ptr::NonNull::<u8>::dangling().as_ptr();
        assert_eq!(zyzyr_decompress(dangling, usize::MAX, out.as_mut_ptr(), 8, &mut n), ZyzyrStatus::LENGTH);
        assert_eq!(zyzyr_decompress(f.as_ptr(), f.len(), dangling, usize::MAX, &mut n), ZyzyrStatus::LENGTH);
        let mut size = 9u64;
        assert_eq!(zyzyr_content_size(ptr::null(), 0, &mut size), ZyzyrStatus::NULL);
        assert_eq!(zyzyr_content_size(f.as_ptr(), f.len(), ptr::null_mut()), ZyzyrStatus::NULL);
        let mut id = 9u32;
        assert_eq!(zyzyr_dictionary_id(ptr::null(), 0, &mut id), ZyzyrStatus::NULL);
        assert_eq!(zyzyr_dictionary_id(f.as_ptr(), f.len(), ptr::null_mut()), ZyzyrStatus::NULL);
        let mut d = ptr::null_mut();
        assert_eq!(zyzyr_dict_new(ptr::null(), 10, &mut d), ZyzyrStatus::NULL);
        assert_eq!(zyzyr_dict_new(f.as_ptr(), f.len(), ptr::null_mut()), ZyzyrStatus::NULL);
        assert_eq!(zyzyr_dict_id(ptr::null()), 0);
        zyzyr_dict_free(ptr::null_mut());
        assert_eq!(zyzyr_decompress_with_dict(f.as_ptr(), f.len(), ptr::null(), out.as_mut_ptr(), 8, &mut n), ZyzyrStatus::NULL);
        let mut p = ptr::null();
        assert_eq!(zyzyr_decoder_decompress(ptr::null_mut(), f.as_ptr(), f.len(), ptr::null(), &mut p, &mut n), ZyzyrStatus::NULL);
        assert_eq!(zyzyr_decoder_decompress_into(ptr::null_mut(), f.as_ptr(), f.len(), ptr::null(), out.as_mut_ptr(), 8, &mut n), ZyzyrStatus::NULL);
        zyzyr_decoder_free(ptr::null_mut());
    }
}

#[test]
fn decode_errors_map_to_codes() {
    let mut out = [0u8; 64];
    let mut n = 0usize;
    let run = |d: &[u8], out: &mut [u8], n: &mut usize| unsafe { code(zyzyr_decompress(d.as_ptr(), d.len(), out.as_mut_ptr(), out.len(), n)) };
    assert_eq!(run(&[], &mut out, &mut n), -10);
    assert_eq!(run(&[1, 2, 3, 4], &mut out, &mut n), -11);
    let mut r = MAGIC.to_vec();
    r.extend_from_slice(&[0x08, 0x00]);
    assert_eq!(run(&r, &mut out, &mut n), -12);
    let mut d = MAGIC.to_vec();
    d.extend_from_slice(&[0x01, 0x00, 7]);
    assert_eq!(run(&d, &mut out, &mut n), -13);
    let mut c = MAGIC.to_vec();
    c.extend_from_slice(&[0x24, 3, ((3 << 3) | 1) as u8, 0, 0, b'a', b'b', b'c', 0, 0, 0, 0]);
    assert_eq!(run(&c, &mut out, &mut n), -24);
    assert_eq!(n, 0);
    let mut f = raw_frame(b"abc", true);
    f[5] = 4;
    assert_eq!(run(&f, &mut out, &mut n), -23);
}

#[test]
fn content_size_and_dictionary_id() {
    let mut two = raw_frame(b"abc", true);
    two.extend_from_slice(&[0x50, 0x2A, 0x4D, 0x18, 2, 0, 0, 0, 1, 2]);
    two.extend_from_slice(&raw_frame(b"defgh", true));
    let mut size = 0u64;
    unsafe {
        assert_eq!(zyzyr_content_size(two.as_ptr(), two.len(), &mut size), ZyzyrStatus::OK);
        assert_eq!(size, 8);
        let open = raw_frame(b"abc", false);
        assert_eq!(zyzyr_content_size(open.as_ptr(), open.len(), &mut size), ZyzyrStatus::UNKNOWN_SIZE);
        assert_eq!(size, 0);
        let cut = &two[..7];
        assert_eq!(code(zyzyr_content_size(cut.as_ptr(), cut.len(), &mut size)), -10);
        let mut id = 1u32;
        assert_eq!(zyzyr_dictionary_id(two.as_ptr(), two.len(), &mut id), ZyzyrStatus::OK);
        assert_eq!(id, 0);
        let mut d = MAGIC.to_vec();
        d.extend_from_slice(&[0x02, 0x00, 0x34, 0x12]);
        assert_eq!(zyzyr_dictionary_id(d.as_ptr(), d.len(), &mut id), ZyzyrStatus::OK);
        assert_eq!(id, 0x1234);
    }
}

#[test]
fn decoder_handle_reuse() {
    let a = raw_frame(b"first frame", true);
    let b = raw_frame(b"second", false);
    unsafe {
        let dec = zyzyr_decoder_new();
        assert!(!dec.is_null());
        for _ in 0..3 {
            for (f, want) in [(&a, &b"first frame"[..]), (&b, &b"second"[..])] {
                let mut p = ptr::null();
                let mut n = 0usize;
                let s = zyzyr_decoder_decompress(dec, f.as_ptr(), f.len(), ptr::null(), &mut p, &mut n);
                assert_eq!(s, ZyzyrStatus::OK);
                assert_eq!(std::slice::from_raw_parts(p, n), want);
                let mut out = [0u8; 16];
                let s = zyzyr_decoder_decompress_into(dec, f.as_ptr(), f.len(), ptr::null(), out.as_mut_ptr(), 16, &mut n);
                assert_eq!(s, ZyzyrStatus::OK);
                assert_eq!(&out[..n], want);
                let s = zyzyr_decoder_decompress_into(dec, f.as_ptr(), f.len(), ptr::null(), out.as_mut_ptr(), 3, &mut n);
                assert_eq!(s, ZyzyrStatus::SHORT_BUFFER);
                assert_eq!(n, want.len());
            }
        }
        let mut p = ptr::null();
        let mut n = 5usize;
        let s = zyzyr_decoder_decompress(dec, [9u8; 3].as_ptr(), 3, ptr::null(), &mut p, &mut n);
        assert_eq!(code(s), -10);
        assert!(p.is_null());
        assert_eq!(n, 0);
        zyzyr_decoder_free(dec);
    }
}

unsafe fn stream_through(s: *mut ZyzyrStream, z: &[u8], chunk: usize, cap: usize) -> (ZyzyrStatus, Vec<u8>) {
    let mut got = Vec::new();
    let mut buf = vec![0u8; cap];
    for c in z.chunks(chunk) {
        let mut off = 0;
        while off < c.len() {
            let mut used = usize::MAX;
            let st = unsafe { zyzyr_stream_feed(s, c[off..].as_ptr(), c.len() - off, &mut used) };
            if st != ZyzyrStatus::OK {
                return (st, got);
            }
            off += used;
            loop {
                let mut n = usize::MAX;
                assert_eq!(unsafe { zyzyr_stream_drain(s, buf.as_mut_ptr(), buf.len(), &mut n) }, ZyzyrStatus::OK);
                if n == 0 {
                    break;
                }
                got.extend_from_slice(&buf[..n]);
            }
            assert_eq!(unsafe { zyzyr_stream_pending(s) }, 0);
        }
    }
    (unsafe { zyzyr_stream_finish(s) }, got)
}

#[test]
fn incremental_stream_api() {
    let mut two = raw_frame(b"first frame", true);
    two.extend_from_slice(&[0x50, 0x2A, 0x4D, 0x18, 2, 0, 0, 0, 1, 2]);
    two.extend_from_slice(&raw_frame(b"second", false));
    unsafe {
        let s = zyzyr_stream_new(ptr::null());
        assert!(!s.is_null());
        for (chunk, cap) in [(1usize, 1usize), (3, 5), (64, 64)] {
            let (st, got) = stream_through(s, &two, chunk, cap);
            assert_eq!(st, ZyzyrStatus::OK);
            assert_eq!(got, b"first framesecond");
        }
        let (st, _) = stream_through(s, &two[..two.len() - 2], 4, 4);
        assert_eq!(code(st), -10);
        assert_eq!(zyzyr_stream_reset(s), ZyzyrStatus::OK);
        let (st, _) = stream_through(s, b"garbage!", 8, 8);
        assert_eq!(code(st), -11);
        let mut used = 0usize;
        assert_eq!(code(zyzyr_stream_feed(s, two.as_ptr(), two.len(), &mut used)), -11);
        assert_eq!(zyzyr_stream_reset(s), ZyzyrStatus::OK);
        assert_eq!(zyzyr_stream_feed(s, ptr::null(), 0, &mut used), ZyzyrStatus::OK);
        assert_eq!(zyzyr_stream_feed(s, ptr::null(), 4, &mut used), ZyzyrStatus::NULL);
        assert_eq!(zyzyr_stream_feed(s, two.as_ptr(), 4, ptr::null_mut()), ZyzyrStatus::NULL);
        assert_eq!(zyzyr_stream_feed(ptr::null_mut(), two.as_ptr(), 4, &mut used), ZyzyrStatus::NULL);
        let mut n = 0usize;
        assert_eq!(zyzyr_stream_drain(ptr::null_mut(), ptr::null_mut(), 0, &mut n), ZyzyrStatus::NULL);
        assert_eq!(zyzyr_stream_finish(ptr::null()), ZyzyrStatus::NULL);
        assert_eq!(zyzyr_stream_pending(ptr::null()), 0);
        zyzyr_stream_free(s);
        zyzyr_stream_free(ptr::null_mut());
        for c in common::cases() {
            let mut dict = ptr::null_mut();
            if let Some(bytes) = &c.dict {
                assert_eq!(zyzyr_dict_new(bytes.as_ptr(), bytes.len(), &mut dict), ZyzyrStatus::OK);
            }
            let s = zyzyr_stream_new(dict);
            for (chunk, cap) in [(777usize, 1000usize), (65536, 131072)] {
                let (st, got) = stream_through(s, &c.data, chunk, cap);
                assert_eq!(st, ZyzyrStatus::OK, "{}", c.name);
                assert!(c.check(&got), "{}", c.name);
            }
            zyzyr_stream_free(s);
            zyzyr_dict_free(dict);
            println!("{}: {} bytes through zyzyr_stream_*", c.name, c.len);
        }
    }
}

#[test]
fn messages_and_version() {
    for c in [0, 1, -1, -2, -3].into_iter().chain(-24..=-10) {
        let m = unsafe { CStr::from_ptr(zyzyr_status_message(c)) };
        assert!(!m.is_empty());
        assert_ne!(m.to_bytes(), b"unknown status code", "code {c}");
    }
    let m = unsafe { CStr::from_ptr(zyzyr_status_message(-99)) };
    assert_eq!(m.to_bytes(), b"unknown status code");
    let v = unsafe { CStr::from_ptr(zyzyr_version()) };
    assert_eq!(v.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}

#[test]
fn dictionary_frames_through_the_c_api() {
    let mut checked = 0;
    for c in common::cases() {
        let Some(bytes) = &c.dict else {
            continue;
        };
        unsafe {
            let mut dict = ptr::null_mut();
            assert_eq!(zyzyr_dict_new(bytes.as_ptr(), bytes.len(), &mut dict), ZyzyrStatus::OK);
            assert!(!dict.is_null());
            let did = zyzyr_dict_id(dict);
            let data = &c.data;
            let mut id = 0u32;
            assert_eq!(zyzyr_dictionary_id(data.as_ptr(), data.len(), &mut id), ZyzyrStatus::OK);
            assert_eq!(id, did);
            let mut out = vec![0u8; c.len];
            let mut n = 0usize;
            let s = zyzyr_decompress(data.as_ptr(), data.len(), out.as_mut_ptr(), out.len(), &mut n);
            assert!(code(s) < 0, "{}", c.name);
            let s = zyzyr_decompress_with_dict(data.as_ptr(), data.len(), dict, out.as_mut_ptr(), out.len(), &mut n);
            assert_eq!(s, ZyzyrStatus::OK, "{}", c.name);
            assert!(c.check(&out[..n]), "{}", c.name);
            let dec = zyzyr_decoder_new();
            let mut q = ptr::null();
            let s = zyzyr_decoder_decompress(dec, data.as_ptr(), data.len(), dict, &mut q, &mut n);
            assert_eq!(s, ZyzyrStatus::OK);
            assert!(c.check(std::slice::from_raw_parts(q, n)));
            zyzyr_decoder_free(dec);
            zyzyr_dict_free(dict);
        }
        checked += 1;
    }
    assert!(checked >= 3);
    println!("{checked} dictionary frames through the C API");
}
