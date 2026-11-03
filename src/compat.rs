#![allow(non_snake_case)]

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_uint, c_ulonglong, c_void, CStr};
use core::ptr;
use core::slice;

use crate::{content_size_first, frame_extent, Decoder, Dictionary, Error, StreamDecoder};

pub const ERR_GENERIC: usize = 1;
pub const ERR_PREFIX_UNKNOWN: usize = 10;
pub const ERR_FRAME_PARAMETER_UNSUPPORTED: usize = 14;
pub const ERR_CORRUPTION_DETECTED: usize = 20;
pub const ERR_CHECKSUM_WRONG: usize = 22;
pub const ERR_DICTIONARY_CORRUPTED: usize = 30;
pub const ERR_DICTIONARY_WRONG: usize = 32;
pub const ERR_PARAMETER_UNSUPPORTED: usize = 40;
pub const ERR_DST_SIZE_TOO_SMALL: usize = 70;
pub const ERR_SRC_SIZE_WRONG: usize = 72;
pub const ERR_DST_BUFFER_NULL: usize = 74;
const ERR_MAX_CODE: usize = 120;

pub const CONTENTSIZE_UNKNOWN: c_ulonglong = !0;
pub const CONTENTSIZE_ERROR: c_ulonglong = !0 - 1;

const D_WINDOW_LOG_MAX: c_int = 100;
const STREAM_IN_SIZE: usize = (128 << 10) + 3;
const STREAM_OUT_SIZE: usize = 128 << 10;

fn err(code: usize) -> usize {
    code.wrapping_neg()
}

fn code_of(e: Error) -> usize {
    err(match e {
        Error::Truncated => ERR_SRC_SIZE_WRONG,
        Error::BadMagic => ERR_PREFIX_UNKNOWN,
        Error::ReservedBit => ERR_FRAME_PARAMETER_UNSUPPORTED,
        Error::Dictionary(_) => ERR_DICTIONARY_WRONG,
        Error::DictionaryFormat => ERR_DICTIONARY_CORRUPTED,
        Error::Checksum => ERR_CHECKSUM_WRONG,
        _ => ERR_CORRUPTION_DETECTED,
    })
}

pub struct ZstdDCtx {
    dec: Decoder,
    dict: Option<Dictionary>,
    scratch: Vec<u8>,
    stream: StreamDecoder,
    held: bool,
}

impl ZstdDCtx {
    fn new() -> Self {
        ZstdDCtx { dec: Decoder::new(), dict: None, scratch: Vec::new(), stream: StreamDecoder::new(), held: false }
    }

    fn reset_stream(&mut self) {
        self.stream.reset();
        self.held = false;
    }

    fn decode(&mut self, input: &[u8], out: &mut Vec<u8>) -> Result<usize, Error> {
        match &self.dict {
            Some(d) => self.dec.decompress_into_with_dict(input, d, out),
            None => self.dec.decompress_into(input, out),
        }
    }
}

#[repr(C)]
pub struct ZstdInBuffer {
    pub src: *const c_void,
    pub size: usize,
    pub pos: usize,
}

#[repr(C)]
pub struct ZstdOutBuffer {
    pub dst: *mut c_void,
    pub size: usize,
    pub pos: usize,
}

unsafe fn bytes<'a>(p: *const c_void, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        return Some(&[]);
    }
    if p.is_null() || len > isize::MAX as usize || (p as usize).checked_add(len).is_none() {
        return None;
    }
    Some(unsafe { slice::from_raw_parts(p.cast::<u8>(), len) })
}

fn dst_ok(dst: *mut c_void, cap: usize) -> Result<(), usize> {
    if dst.is_null() && cap != 0 {
        return Err(err(ERR_DST_BUFFER_NULL));
    }
    if cap > isize::MAX as usize || (dst as usize).checked_add(cap).is_none() {
        return Err(err(ERR_GENERIC));
    }
    Ok(())
}

unsafe fn deliver(src: &[u8], dst: *mut c_void, cap: usize) -> usize {
    if src.len() > cap {
        return err(ERR_DST_SIZE_TOO_SMALL);
    }
    if !src.is_empty() {
        unsafe { ptr::copy_nonoverlapping(src.as_ptr(), dst.cast::<u8>(), src.len()) };
    }
    src.len()
}

unsafe fn one_shot(d: &mut ZstdDCtx, dst: *mut c_void, cap: usize, src: *const c_void, len: usize) -> usize {
    if let Err(e) = dst_ok(dst, cap) {
        return e;
    }
    let Some(input) = (unsafe { bytes(src, len) }) else {
        return err(ERR_GENERIC);
    };
    let mut out = core::mem::take(&mut d.scratch);
    out.clear();
    let r = match d.decode(input, &mut out) {
        Ok(_) => unsafe { deliver(&out, dst, cap) },
        Err(e) => code_of(e),
    };
    d.scratch = out;
    r
}

#[no_mangle]
pub unsafe extern "C" fn ZSTD_decompress(dst: *mut c_void, dst_capacity: usize, src: *const c_void, compressed_size: usize) -> usize {
    let mut d = ZstdDCtx::new();
    unsafe { one_shot(&mut d, dst, dst_capacity, src, compressed_size) }
}

#[no_mangle]
pub unsafe extern "C" fn ZSTD_getFrameContentSize(src: *const c_void, src_size: usize) -> c_ulonglong {
    let Some(input) = (unsafe { bytes(src, src_size) }) else {
        return CONTENTSIZE_ERROR;
    };
    match content_size_first(input) {
        Ok(Some(n)) => n,
        Ok(None) => CONTENTSIZE_UNKNOWN,
        Err(_) => CONTENTSIZE_ERROR,
    }
}

#[no_mangle]
pub unsafe extern "C" fn ZSTD_findFrameCompressedSize(src: *const c_void, src_size: usize) -> usize {
    let Some(input) = (unsafe { bytes(src, src_size) }) else {
        return err(ERR_GENERIC);
    };
    match frame_extent(input) {
        Ok(Some(n)) => n,
        Ok(None) => err(ERR_SRC_SIZE_WRONG),
        Err(e) => code_of(e),
    }
}

#[no_mangle]
pub extern "C" fn ZSTD_isError(code: usize) -> c_uint {
    (code > err(ERR_MAX_CODE)) as c_uint
}

#[no_mangle]
pub extern "C" fn ZSTD_getErrorName(code: usize) -> *const c_char {
    let text: &CStr = if code <= err(ERR_MAX_CODE) {
        c"No error detected"
    } else {
        match code.wrapping_neg() {
            ERR_GENERIC => c"Error (generic)",
            ERR_PREFIX_UNKNOWN => c"Unknown frame descriptor",
            ERR_FRAME_PARAMETER_UNSUPPORTED => c"Unsupported frame parameter",
            ERR_CORRUPTION_DETECTED => c"Data corruption detected",
            ERR_CHECKSUM_WRONG => c"Restored data doesn't match checksum",
            ERR_DICTIONARY_CORRUPTED => c"Dictionary is corrupted",
            ERR_DICTIONARY_WRONG => c"Dictionary mismatch",
            ERR_PARAMETER_UNSUPPORTED => c"Unsupported parameter",
            ERR_DST_SIZE_TOO_SMALL => c"Destination buffer is too small",
            ERR_SRC_SIZE_WRONG => c"Src size is incorrect",
            ERR_DST_BUFFER_NULL => c"Operation on NULL destination buffer",
            _ => c"Unspecified error code",
        }
    };
    text.as_ptr()
}

#[no_mangle]
pub extern "C" fn ZSTD_createDCtx() -> *mut ZstdDCtx {
    Box::into_raw(Box::new(ZstdDCtx::new()))
}

#[no_mangle]
pub unsafe extern "C" fn ZSTD_freeDCtx(dctx: *mut ZstdDCtx) -> usize {
    if !dctx.is_null() {
        unsafe { drop(Box::from_raw(dctx)) };
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn ZSTD_decompressDCtx(dctx: *mut ZstdDCtx, dst: *mut c_void, dst_capacity: usize, src: *const c_void, src_size: usize) -> usize {
    match unsafe { dctx.as_mut() } {
        Some(d) => unsafe { one_shot(d, dst, dst_capacity, src, src_size) },
        None => err(ERR_GENERIC),
    }
}

#[no_mangle]
pub unsafe extern "C" fn ZSTD_decompress_usingDict(dctx: *mut ZstdDCtx, dst: *mut c_void, dst_capacity: usize, src: *const c_void, src_size: usize, dict: *const c_void, dict_size: usize) -> usize {
    let Some(d) = (unsafe { dctx.as_mut() }) else {
        return err(ERR_GENERIC);
    };
    let Some(bytes) = (unsafe { bytes(dict, dict_size) }) else {
        return err(ERR_GENERIC);
    };
    let saved = d.dict.take();
    if !bytes.is_empty() {
        match Dictionary::new(bytes) {
            Ok(k) => d.dict = Some(k),
            Err(e) => {
                d.dict = saved;
                return code_of(e);
            }
        }
    }
    let r = unsafe { one_shot(d, dst, dst_capacity, src, src_size) };
    d.dict = saved;
    r
}

#[no_mangle]
pub unsafe extern "C" fn ZSTD_DCtx_loadDictionary(dctx: *mut ZstdDCtx, dict: *const c_void, dict_size: usize) -> usize {
    let Some(d) = (unsafe { dctx.as_mut() }) else {
        return err(ERR_GENERIC);
    };
    let Some(bytes) = (unsafe { bytes(dict, dict_size) }) else {
        return err(ERR_GENERIC);
    };
    if bytes.is_empty() {
        d.dict = None;
        return 0;
    }
    match Dictionary::new(bytes) {
        Ok(k) => {
            d.dict = Some(k);
            0
        }
        Err(e) => code_of(e),
    }
}

#[no_mangle]
pub unsafe extern "C" fn ZSTD_DCtx_setParameter(dctx: *mut ZstdDCtx, param: c_int, value: c_int) -> usize {
    if dctx.is_null() {
        return err(ERR_GENERIC);
    }
    match param {
        D_WINDOW_LOG_MAX if value == 0 || (10..=31).contains(&value) => 0,
        _ => err(ERR_PARAMETER_UNSUPPORTED),
    }
}

#[no_mangle]
pub unsafe extern "C" fn ZSTD_DCtx_reset(dctx: *mut ZstdDCtx, reset: c_int) -> usize {
    let Some(d) = (unsafe { dctx.as_mut() }) else {
        return err(ERR_GENERIC);
    };
    if !(1..=3).contains(&reset) {
        return err(ERR_PARAMETER_UNSUPPORTED);
    }
    d.reset_stream();
    if reset >= 2 {
        d.dict = None;
    }
    0
}

#[no_mangle]
pub extern "C" fn ZSTD_createDStream() -> *mut ZstdDCtx {
    ZSTD_createDCtx()
}

#[no_mangle]
pub unsafe extern "C" fn ZSTD_freeDStream(zds: *mut ZstdDCtx) -> usize {
    unsafe { ZSTD_freeDCtx(zds) }
}

#[no_mangle]
pub unsafe extern "C" fn ZSTD_initDStream(zds: *mut ZstdDCtx) -> usize {
    let Some(d) = (unsafe { zds.as_mut() }) else {
        return err(ERR_GENERIC);
    };
    d.reset_stream();
    STREAM_IN_SIZE
}

#[no_mangle]
pub extern "C" fn ZSTD_DStreamInSize() -> usize {
    STREAM_IN_SIZE
}

#[no_mangle]
pub extern "C" fn ZSTD_DStreamOutSize() -> usize {
    STREAM_OUT_SIZE
}

unsafe fn flush(d: &mut ZstdDCtx, o: &mut ZstdOutBuffer) {
    let avail = d.stream.pending();
    let n = avail.len().min(o.size - o.pos);
    if n != 0 {
        unsafe { ptr::copy_nonoverlapping(avail.as_ptr(), o.dst.cast::<u8>().add(o.pos), n) };
        o.pos += n;
        d.stream.consume(n);
    }
}

#[no_mangle]
pub unsafe extern "C" fn ZSTD_decompressStream(zds: *mut ZstdDCtx, output: *mut ZstdOutBuffer, input: *mut ZstdInBuffer) -> usize {
    let (Some(d), Some(o), Some(i)) = (unsafe { zds.as_mut() }, unsafe { output.as_mut() }, unsafe { input.as_mut() }) else {
        return err(ERR_GENERIC);
    };
    if i.pos > i.size || o.pos > o.size {
        return err(ERR_GENERIC);
    }
    if let Err(e) = dst_ok(o.dst, o.size) {
        return e;
    }
    let Some(whole) = (unsafe { bytes(i.src, i.size) }) else {
        return err(ERR_GENERIC);
    };
    unsafe { flush(d, o) };
    if !d.stream.pending().is_empty() {
        return d.stream.pending().len();
    }
    if d.held {
        if i.pos < i.size {
            i.pos += 1;
        }
        d.held = false;
    }
    let start = i.pos;
    while i.pos < i.size {
        let n = match d.stream.feed_inner(&whole[i.pos..], d.dict.as_ref()) {
            Ok(n) => n,
            Err(e) => {
                d.reset_stream();
                return code_of(e);
            }
        };
        i.pos += n;
        unsafe { flush(d, o) };
        let left = d.stream.pending().len();
        if left != 0 {
            if i.pos == i.size && i.pos > start {
                i.pos -= 1;
                d.held = true;
            }
            return left;
        }
        if n == 0 {
            break;
        }
    }
    if d.stream.at_frame_boundary() {
        0
    } else {
        d.stream.input_hint().max(1)
    }
}
