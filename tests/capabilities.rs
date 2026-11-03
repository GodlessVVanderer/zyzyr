mod common;

use std::ffi::CStr;

const MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];
const SKIP_MAGIC: [u8; 4] = [0x50, 0x2A, 0x4D, 0x18];

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

fn skip_frame(payload: &[u8]) -> Vec<u8> {
    let mut v = SKIP_MAGIC.to_vec();
    v.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    v.extend_from_slice(payload);
    v
}

// --- One-shot API ---

#[test]
fn decompress_returns_owned_output() {
    let frame = raw_frame(b"owned allocation", true);
    let out = zyzyr::decompress(&frame).unwrap();
    assert_eq!(out, b"owned allocation");
}

#[test]
fn decompress_into_appends_to_existing_data() {
    let frame = raw_frame(b"second", true);
    let mut buf = b"first|".to_vec();
    let n = zyzyr::decompress_into(&frame, &mut buf).unwrap();
    assert_eq!(n, 6);
    assert_eq!(buf, b"first|second");
}

#[test]
fn concatenated_frames_decompress_as_one_sequence() {
    let mut cat = raw_frame(b"hello", true);
    cat.extend_from_slice(&raw_frame(b" world", true));
    let out = zyzyr::decompress(&cat).unwrap();
    assert_eq!(out, b"hello world");
}

#[test]
fn skippable_frames_are_transparent_in_concatenation() {
    let mut cat = raw_frame(b"before", true);
    cat.extend_from_slice(&skip_frame(b"metadata nobody reads"));
    cat.extend_from_slice(&raw_frame(b"|after", true));
    let out = zyzyr::decompress(&cat).unwrap();
    assert_eq!(out, b"before|after");
}

// --- Content size ---

#[test]
fn content_size_sums_across_concatenated_frames() {
    let mut cat = raw_frame(b"abc", true);
    cat.extend_from_slice(&raw_frame(b"defgh", true));
    assert_eq!(zyzyr::content_size(&cat).unwrap(), Some(8));
}

#[test]
fn content_size_is_none_when_any_frame_omits_it() {
    let mut cat = raw_frame(b"abc", true);
    cat.extend_from_slice(&raw_frame(b"def", false));
    assert_eq!(zyzyr::content_size(&cat).unwrap(), None);
}

#[test]
fn content_size_skips_over_skippable_frames() {
    let mut cat = raw_frame(b"abc", true);
    cat.extend_from_slice(&skip_frame(b"skip me"));
    cat.extend_from_slice(&raw_frame(b"de", true));
    assert_eq!(zyzyr::content_size(&cat).unwrap(), Some(5));
}

// --- Dictionary ID ---

#[test]
fn dictionary_id_reads_the_first_frame() {
    let mut frame = MAGIC.to_vec();
    frame.extend_from_slice(&[0x02, 0x00, 0x34, 0x12]);
    assert_eq!(zyzyr::dictionary_id(&frame).unwrap(), 0x1234);
}

#[test]
fn dictionary_id_is_zero_when_absent() {
    let frame = raw_frame(b"no dict", true);
    assert_eq!(zyzyr::dictionary_id(&frame).unwrap(), 0);
}

// --- Reusable Decoder ---

#[test]
fn decoder_reuses_context_across_many_frames() {
    let mut dec = zyzyr::Decoder::new();
    for i in 0u32..100 {
        let payload = format!("frame {i}");
        let frame = raw_frame(payload.as_bytes(), true);
        let out = dec.decompress(&frame).unwrap();
        assert_eq!(out, payload.as_bytes());
    }
}

#[test]
fn decoder_recovers_after_error() {
    let mut dec = zyzyr::Decoder::new();
    assert!(dec.decompress(&[0xFF, 0xFF]).is_err());
    let frame = raw_frame(b"still works", true);
    assert_eq!(dec.decompress(&frame).unwrap(), b"still works");
}

#[test]
fn decoder_into_appends_like_free_function() {
    let mut dec = zyzyr::Decoder::new();
    let mut buf = b"pre|".to_vec();
    let frame = raw_frame(b"post", true);
    dec.decompress_into(&frame, &mut buf).unwrap();
    assert_eq!(buf, b"pre|post");
}

// --- Streaming ---

#[test]
fn stream_at_frame_boundary_tracks_position() {
    let mut s = zyzyr::StreamDecoder::new();
    assert!(s.at_frame_boundary());
    let frame = raw_frame(b"hello", true);
    s.feed(&frame).unwrap();
    let mut out = vec![0u8; 64];
    s.drain(&mut out);
    assert!(s.at_frame_boundary());
}

#[test]
fn stream_input_hint_guides_feeding() {
    let mut s = zyzyr::StreamDecoder::new();
    let frame = raw_frame(b"test", true);
    let hint0 = s.input_hint();
    assert_eq!(hint0, 0);
    s.feed(&frame[..4]).unwrap();
    let hint1 = s.input_hint();
    assert!(hint1 > 0);
}

#[test]
fn stream_memory_reports_allocation() {
    let mut s = zyzyr::StreamDecoder::new();
    let m0 = s.memory();
    let frame = raw_frame(b"data that needs a window", true);
    s.feed(&frame).unwrap();
    let m1 = s.memory();
    assert!(m1 >= m0);
}

#[test]
fn stream_pending_gives_direct_access() {
    let mut s = zyzyr::StreamDecoder::new();
    let frame = raw_frame(b"direct", true);
    s.feed(&frame).unwrap();
    let pending = s.pending();
    assert_eq!(pending, b"direct");
    s.consume(3);
    assert_eq!(s.pending(), b"ect");
}

#[test]
fn stream_reset_clears_state_for_reuse() {
    let mut s = zyzyr::StreamDecoder::new();
    let frame = raw_frame(b"first", true);
    s.feed(&frame).unwrap();
    let mut out = vec![0u8; 64];
    s.drain(&mut out);
    s.reset();
    assert!(s.at_frame_boundary());
    let frame2 = raw_frame(b"second", true);
    s.feed(&frame2).unwrap();
    let n = s.drain(&mut out);
    assert_eq!(&out[..n], b"second");
}

// --- Error reporting ---

#[test]
fn every_error_variant_has_a_nonempty_c_string() {
    let errors = [
        zyzyr::Error::Truncated,
        zyzyr::Error::BadMagic,
        zyzyr::Error::ReservedBit,
        zyzyr::Error::Dictionary(0),
        zyzyr::Error::DictionaryFormat,
        zyzyr::Error::BlockType,
        zyzyr::Error::BlockSize,
        zyzyr::Error::Literals,
        zyzyr::Error::Huffman,
        zyzyr::Error::Fse,
        zyzyr::Error::Bitstream,
        zyzyr::Error::Sequences,
        zyzyr::Error::Offset,
        zyzyr::Error::ContentSize,
        zyzyr::Error::Checksum,
    ];
    for e in errors {
        let c: &CStr = e.text();
        assert!(!c.is_empty(), "{e:?} has empty text");
        assert!(c.to_str().is_ok(), "{e:?} text is not valid UTF-8");
    }
}

#[test]
fn error_display_matches_text_for_non_dictionary() {
    let e = zyzyr::Error::Truncated;
    let display = format!("{e}");
    let text = e.text().to_str().unwrap();
    assert_eq!(display, text);
}

#[test]
fn error_implements_std_error() {
    fn assert_std_error<T: std::error::Error>() {}
    assert_std_error::<zyzyr::Error>();
}

// --- Dictionary (Rust API) ---

#[test]
fn dictionary_reports_its_id() {
    for c in common::cases() {
        let Some(dict_bytes) = &c.dict else { continue };
        let dict = zyzyr::Dictionary::new(dict_bytes).unwrap();
        let id = dict.id();
        let frame_id = zyzyr::dictionary_id(&c.data).unwrap();
        assert_eq!(id, frame_id, "{}", c.name);
    }
}

#[test]
fn raw_content_dictionary_has_id_zero() {
    let raw_dict = b"this is just raw content, no zstd dict magic";
    let dict = zyzyr::Dictionary::new(raw_dict).unwrap();
    assert_eq!(dict.id(), 0);
}

// --- Default implementations ---

#[test]
fn decoder_implements_default() {
    let _d: zyzyr::Decoder = Default::default();
}

#[test]
fn stream_decoder_implements_default() {
    let _s: zyzyr::StreamDecoder = Default::default();
}

// --- Large concatenation ---

#[test]
fn thousand_frame_concatenation() {
    let mut cat = Vec::new();
    for i in 0u16..1000 {
        cat.extend_from_slice(&raw_frame(&i.to_le_bytes(), true));
    }
    let out = zyzyr::decompress(&cat).unwrap();
    assert_eq!(out.len(), 2000);
    for (i, chunk) in out.chunks_exact(2).enumerate() {
        assert_eq!(chunk, &(i as u16).to_le_bytes(), "frame {i}");
    }
}

// --- Empty frames ---

#[test]
fn empty_frame_is_valid() {
    let frame = raw_frame(b"", true);
    let out = zyzyr::decompress(&frame).unwrap();
    assert!(out.is_empty());
}

#[test]
fn empty_skippable_frame_alone() {
    let frame = skip_frame(b"");
    let out = zyzyr::decompress(&frame).unwrap();
    assert!(out.is_empty());
}
