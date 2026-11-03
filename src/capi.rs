use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::{c_char, CStr};
use core::ptr;
use core::slice;

use crate::{Decoder, Dictionary, Error, StreamDecoder};

pub struct ZyzyrDecoder {
    inner: Decoder,
    out: Vec<u8>,
}

pub struct ZyzyrDict {
    inner: Dictionary,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ZyzyrStatus {
    pub code: i32,
}

impl ZyzyrStatus {
    pub const OK: Self = Self { code: 0 };
    pub const UNKNOWN_SIZE: Self = Self { code: 1 };
    pub const NULL: Self = Self { code: -1 };
    pub const SHORT_BUFFER: Self = Self { code: -2 };
    pub const LENGTH: Self = Self { code: -3 };
}

const ERRORS: [(Error, i32); 15] = [
    (Error::Truncated, -10),
    (Error::BadMagic, -11),
    (Error::ReservedBit, -12),
    (Error::Dictionary(0), -13),
    (Error::DictionaryFormat, -14),
    (Error::BlockType, -15),
    (Error::BlockSize, -16),
    (Error::Literals, -17),
    (Error::Huffman, -18),
    (Error::Fse, -19),
    (Error::Bitstream, -20),
    (Error::Sequences, -21),
    (Error::Offset, -22),
    (Error::ContentSize, -23),
    (Error::Checksum, -24),
];

impl From<Error> for ZyzyrStatus {
    fn from(e: Error) -> Self {
        let e = match e {
            Error::Dictionary(_) => Error::Dictionary(0),
            e => e,
        };
        let code = ERRORS.iter().find(|(k, _)| *k == e).map_or(-10, |&(_, c)| c);
        ZyzyrStatus { code }
    }
}

unsafe fn input<'a>(data: *const u8, len: usize) -> Result<&'a [u8], ZyzyrStatus> {
    if data.is_null() {
        return Err(ZyzyrStatus::NULL);
    }
    if len > isize::MAX as usize || (data as usize).checked_add(len).is_none() {
        return Err(ZyzyrStatus::LENGTH);
    }
    Ok(unsafe { slice::from_raw_parts(data, len) })
}

unsafe fn check_out(out: *mut u8, out_cap: usize, out_len: *mut usize) -> Result<(), ZyzyrStatus> {
    if out_len.is_null() || (out.is_null() && out_cap != 0) {
        return Err(ZyzyrStatus::NULL);
    }
    if out_cap > isize::MAX as usize || (out as usize).checked_add(out_cap).is_none() {
        return Err(ZyzyrStatus::LENGTH);
    }
    Ok(())
}

unsafe fn deliver(src: &[u8], out: *mut u8, out_cap: usize, out_len: *mut usize) -> ZyzyrStatus {
    unsafe { *out_len = src.len() };
    if src.len() > out_cap {
        return ZyzyrStatus::SHORT_BUFFER;
    }
    if !src.is_empty() {
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), out, src.len()) };
    }
    ZyzyrStatus::OK
}

unsafe fn one_shot(dec: &mut Decoder, scratch: &mut Vec<u8>, data: *const u8, len: usize, dict: Option<&Dictionary>, out: *mut u8, out_cap: usize, out_len: *mut usize) -> ZyzyrStatus {
    if let Err(s) = unsafe { check_out(out, out_cap, out_len) } {
        return s;
    }
    unsafe { *out_len = 0 };
    let src = match unsafe { input(data, len) } {
        Ok(s) => s,
        Err(s) => return s,
    };
    scratch.clear();
    let r = match dict {
        Some(d) => dec.decompress_into_with_dict(src, d, scratch),
        None => dec.decompress_into(src, scratch),
    };
    match r {
        Ok(_) => unsafe { deliver(scratch, out, out_cap, out_len) },
        Err(e) => e.into(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_decompress(data: *const u8, len: usize, out: *mut u8, out_cap: usize, out_len: *mut usize) -> ZyzyrStatus {
    let mut scratch = Vec::new();
    unsafe { one_shot(&mut Decoder::new(), &mut scratch, data, len, None, out, out_cap, out_len) }
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_decompress_with_dict(data: *const u8, len: usize, dict: *const ZyzyrDict, out: *mut u8, out_cap: usize, out_len: *mut usize) -> ZyzyrStatus {
    let Some(d) = (unsafe { dict.as_ref() }) else {
        return ZyzyrStatus::NULL;
    };
    let mut scratch = Vec::new();
    unsafe { one_shot(&mut Decoder::new(), &mut scratch, data, len, Some(&d.inner), out, out_cap, out_len) }
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_content_size(data: *const u8, len: usize, size: *mut u64) -> ZyzyrStatus {
    if size.is_null() {
        return ZyzyrStatus::NULL;
    }
    unsafe { *size = 0 };
    let src = match unsafe { input(data, len) } {
        Ok(s) => s,
        Err(s) => return s,
    };
    match crate::content_size(src) {
        Ok(Some(n)) => {
            unsafe { *size = n };
            ZyzyrStatus::OK
        }
        Ok(None) => ZyzyrStatus::UNKNOWN_SIZE,
        Err(e) => e.into(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_dictionary_id(data: *const u8, len: usize, id: *mut u32) -> ZyzyrStatus {
    if id.is_null() {
        return ZyzyrStatus::NULL;
    }
    unsafe { *id = 0 };
    let src = match unsafe { input(data, len) } {
        Ok(s) => s,
        Err(s) => return s,
    };
    match crate::dictionary_id(src) {
        Ok(v) => {
            unsafe { *id = v };
            ZyzyrStatus::OK
        }
        Err(e) => e.into(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_dict_new(data: *const u8, len: usize, out: *mut *mut ZyzyrDict) -> ZyzyrStatus {
    if out.is_null() {
        return ZyzyrStatus::NULL;
    }
    unsafe { *out = ptr::null_mut() };
    let src = match unsafe { input(data, len) } {
        Ok(s) => s,
        Err(s) => return s,
    };
    match Dictionary::new(src) {
        Ok(inner) => {
            unsafe { *out = Box::into_raw(Box::new(ZyzyrDict { inner })) };
            ZyzyrStatus::OK
        }
        Err(e) => e.into(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_dict_free(dict: *mut ZyzyrDict) {
    if !dict.is_null() {
        unsafe { drop(Box::from_raw(dict)) }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_dict_id(dict: *const ZyzyrDict) -> u32 {
    match unsafe { dict.as_ref() } {
        Some(d) => d.inner.id(),
        None => 0,
    }
}

#[no_mangle]
pub extern "C" fn zyzyr_decoder_new() -> *mut ZyzyrDecoder {
    Box::into_raw(Box::new(ZyzyrDecoder { inner: Decoder::new(), out: Vec::new() }))
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_decoder_free(dec: *mut ZyzyrDecoder) {
    if !dec.is_null() {
        unsafe { drop(Box::from_raw(dec)) }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_decoder_decompress(dec: *mut ZyzyrDecoder, data: *const u8, len: usize, dict: *const ZyzyrDict, out: *mut *const u8, out_len: *mut usize) -> ZyzyrStatus {
    let Some(d) = (unsafe { dec.as_mut() }) else {
        return ZyzyrStatus::NULL;
    };
    if out.is_null() || out_len.is_null() {
        return ZyzyrStatus::NULL;
    }
    unsafe {
        *out = ptr::null();
        *out_len = 0;
    }
    let src = match unsafe { input(data, len) } {
        Ok(s) => s,
        Err(s) => return s,
    };
    d.out.clear();
    let r = match unsafe { dict.as_ref() } {
        Some(k) => d.inner.decompress_into_with_dict(src, &k.inner, &mut d.out),
        None => d.inner.decompress_into(src, &mut d.out),
    };
    match r {
        Ok(n) => {
            unsafe {
                *out = d.out.as_ptr();
                *out_len = n;
            }
            ZyzyrStatus::OK
        }
        Err(e) => e.into(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_decoder_decompress_into(dec: *mut ZyzyrDecoder, data: *const u8, len: usize, dict: *const ZyzyrDict, out: *mut u8, out_cap: usize, out_len: *mut usize) -> ZyzyrStatus {
    let Some(d) = (unsafe { dec.as_mut() }) else {
        return ZyzyrStatus::NULL;
    };
    let k = unsafe { dict.as_ref() }.map(|k| &k.inner);
    unsafe { one_shot(&mut d.inner, &mut d.out, data, len, k, out, out_cap, out_len) }
}

pub struct ZyzyrStream {
    inner: StreamDecoder,
    dict: *const ZyzyrDict,
}

#[no_mangle]
pub extern "C" fn zyzyr_stream_new(dict: *const ZyzyrDict) -> *mut ZyzyrStream {
    Box::into_raw(Box::new(ZyzyrStream { inner: StreamDecoder::new(), dict }))
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_stream_free(stream: *mut ZyzyrStream) {
    if !stream.is_null() {
        unsafe { drop(Box::from_raw(stream)) }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_stream_feed(stream: *mut ZyzyrStream, data: *const u8, len: usize, consumed: *mut usize) -> ZyzyrStatus {
    let Some(s) = (unsafe { stream.as_mut() }) else {
        return ZyzyrStatus::NULL;
    };
    if consumed.is_null() {
        return ZyzyrStatus::NULL;
    }
    unsafe { *consumed = 0 };
    let src = if len == 0 {
        &[][..]
    } else {
        match unsafe { input(data, len) } {
            Ok(x) => x,
            Err(e) => return e,
        }
    };
    let dict = unsafe { s.dict.as_ref() }.map(|d| &d.inner);
    match s.inner.feed_inner(src, dict) {
        Ok(n) => {
            unsafe { *consumed = n };
            ZyzyrStatus::OK
        }
        Err(e) => e.into(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_stream_drain(stream: *mut ZyzyrStream, out: *mut u8, out_cap: usize, written: *mut usize) -> ZyzyrStatus {
    let Some(s) = (unsafe { stream.as_mut() }) else {
        return ZyzyrStatus::NULL;
    };
    if let Err(e) = unsafe { check_out(out, out_cap, written) } {
        return e;
    }
    let avail = s.inner.pending();
    let n = avail.len().min(out_cap);
    if n != 0 {
        unsafe { ptr::copy_nonoverlapping(avail.as_ptr(), out, n) };
    }
    s.inner.consume(n);
    unsafe { *written = n };
    ZyzyrStatus::OK
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_stream_pending(stream: *const ZyzyrStream) -> usize {
    match unsafe { stream.as_ref() } {
        Some(s) => s.inner.pending().len(),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_stream_finish(stream: *const ZyzyrStream) -> ZyzyrStatus {
    match unsafe { stream.as_ref() } {
        Some(s) => match s.inner.finish() {
            Ok(()) => ZyzyrStatus::OK,
            Err(e) => e.into(),
        },
        None => ZyzyrStatus::NULL,
    }
}

#[no_mangle]
pub unsafe extern "C" fn zyzyr_stream_reset(stream: *mut ZyzyrStream) -> ZyzyrStatus {
    match unsafe { stream.as_mut() } {
        Some(s) => {
            s.inner.reset();
            ZyzyrStatus::OK
        }
        None => ZyzyrStatus::NULL,
    }
}

#[no_mangle]
pub extern "C" fn zyzyr_status_message(code: i32) -> *const c_char {
    let text: &CStr = match code {
        0 => c"ok",
        1 => c"a frame does not record its content size",
        -1 => c"null pointer argument",
        -2 => c"output buffer too small",
        -3 => c"length outside the addressable range",
        c => match ERRORS.iter().find(|&&(_, k)| k == c) {
            Some(&(e, _)) => e.text(),
            None => c"unknown status code",
        },
    };
    text.as_ptr()
}

#[no_mangle]
pub extern "C" fn zyzyr_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}
