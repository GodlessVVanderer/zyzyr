use std::panic;

fn must_not_panic(name: &str, data: &[u8]) {
    let r = panic::catch_unwind(|| {
        let _ = zyzyr::decompress(data);
    });
    assert!(r.is_ok(), "{name}: panicked");
}

fn must_not_panic_stream(name: &str, data: &[u8]) {
    let r = panic::catch_unwind(|| {
        let mut s = zyzyr::StreamDecoder::new();
        let mut out = vec![0u8; 1 << 16];
        let chunk_sizes = [1, 3, 7, 13, 64, 256, 1024, 4096];
        let mut pos = 0;
        for &cs in chunk_sizes.iter().cycle() {
            if pos >= data.len() {
                break;
            }
            let end = (pos + cs).min(data.len());
            match s.feed(&data[pos..end]) {
                Ok(consumed) => pos += consumed,
                Err(_) => break,
            }
            let _ = s.drain(&mut out);
        }
        let _ = s.finish();
    });
    assert!(r.is_ok(), "{name}: stream panicked");
}

#[test]
fn empty_input() {
    must_not_panic("empty", &[]);
    must_not_panic_stream("empty-stream", &[]);
}

#[test]
fn single_bytes() {
    for b in 0..=255u8 {
        must_not_panic(&format!("byte-{b:#04x}"), &[b]);
    }
}

#[test]
fn two_byte_combos() {
    for hi in [0x00, 0x28, 0xFD, 0xFF] {
        for lo in 0..=255u8 {
            must_not_panic(&format!("pair-{hi:#04x}-{lo:#04x}"), &[hi, lo]);
        }
    }
}

#[test]
fn zstd_magic_then_garbage() {
    let magic = [0x28, 0xB5, 0x2F, 0xFD];
    must_not_panic("magic-only", &magic);
    must_not_panic("magic+0", &[0x28, 0xB5, 0x2F, 0xFD, 0x00]);
    must_not_panic("magic+ff", &[0x28, 0xB5, 0x2F, 0xFD, 0xFF]);

    for fhd in 0..=255u8 {
        let mut frame = magic.to_vec();
        frame.push(fhd);
        must_not_panic(&format!("magic+fhd-{fhd:#04x}"), &frame);
    }

    for extra in 0..64 {
        let mut frame = magic.to_vec();
        frame.extend(vec![0xAA; extra]);
        must_not_panic(&format!("magic+{extra}xAA"), &frame);
    }
}

#[test]
fn skippable_frame_variants() {
    for tag in 0x184D2A50u32..=0x184D2A5Fu32 {
        let t = tag.to_le_bytes();
        must_not_panic(&format!("skip-{tag:#010x}-trunc"), &t);

        let mut f = t.to_vec();
        f.extend(&[0x00, 0x00, 0x00, 0x00]);
        must_not_panic(&format!("skip-{tag:#010x}-empty"), &f);

        let mut f = t.to_vec();
        f.extend(&[0x04, 0x00, 0x00, 0x00, 0xDE, 0xAD, 0xBE, 0xEF]);
        must_not_panic(&format!("skip-{tag:#010x}-4b"), &f);

        let mut f = t.to_vec();
        f.extend(&[0xFF, 0xFF, 0xFF, 0x7F]);
        must_not_panic(&format!("skip-{tag:#010x}-huge-size"), &f);
    }
}

#[test]
fn decompression_bomb_header() {
    let mut frame = vec![0x28, 0xB5, 0x2F, 0xFD];
    frame.push(0x60 | 0x04);
    frame.extend(&[0xFF; 8]);
    must_not_panic("bomb-header", &frame);
}

#[test]
fn content_size_lies() {
    let mut frame = vec![0x28, 0xB5, 0x2F, 0xFD];
    frame.push(0x20);
    frame.push(0x00);
    frame.extend(&[0x00; 3]);
    must_not_panic("size-lie-tiny", &frame);

    let mut frame = vec![0x28, 0xB5, 0x2F, 0xFD];
    frame.push(0x60);
    frame.extend(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F]);
    frame.extend(&[0x00; 3]);
    must_not_panic("size-lie-huge", &frame);
}

#[test]
fn repeated_magic_no_frames() {
    let magic = [0x28, 0xB5, 0x2F, 0xFD];
    let mut data = Vec::new();
    for _ in 0..100 {
        data.extend_from_slice(&magic);
    }
    must_not_panic("100x-magic", &data);
}

#[test]
fn all_zeros() {
    for len in [0, 1, 4, 8, 16, 64, 256, 1024, 4096, 65536] {
        must_not_panic(&format!("zeros-{len}"), &vec![0u8; len]);
        must_not_panic_stream(&format!("zeros-stream-{len}"), &vec![0u8; len]);
    }
}

#[test]
fn all_ones() {
    for len in [1, 4, 8, 16, 64, 256, 1024, 4096, 65536] {
        must_not_panic(&format!("ones-{len}"), &vec![0xFF; len]);
        must_not_panic_stream(&format!("ones-stream-{len}"), &vec![0xFF; len]);
    }
}

#[test]
fn stream_tiny_chunks() {
    let magic = [0x28, 0xB5, 0x2F, 0xFD];
    let mut data = magic.to_vec();
    data.extend(&[0x00; 32]);
    must_not_panic_stream("stream-tiny", &data);
}

#[test]
fn dictionary_id_on_garbage() {
    assert!(zyzyr::dictionary_id(&[]).is_err());
    assert!(zyzyr::dictionary_id(&[0x00]).is_err());
    assert!(zyzyr::dictionary_id(&[0x28, 0xB5, 0x2F, 0xFD]).is_err());
    assert!(zyzyr::dictionary_id(&[0xFF; 100]).is_err());
}

#[test]
fn content_size_on_garbage() {
    assert!(zyzyr::content_size(&[]).is_err());
    assert!(zyzyr::content_size(&[0x00]).is_err());
    assert!(zyzyr::content_size(&[0x28, 0xB5, 0x2F, 0xFD]).is_err());
}

#[test]
fn heavy_mutation_stress() {
    let valid_frames: Vec<Vec<u8>> = std::fs::read_dir("tests/data")
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "zst"))
        .filter_map(|e| std::fs::read(e.path()).ok())
        .collect();

    let mut s = 0xDEAD_BEEF_CAFE_BABEu64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };

    let mut cases = 0usize;
    for frame in &valid_frames {
        for _ in 0..2000 {
            let mut m = frame.clone();
            let mutations = 1 + next() % 8;
            for _ in 0..mutations {
                if m.is_empty() {
                    break;
                }
                match next() % 6 {
                    0 => {
                        let p = (next() % m.len() as u64) as usize;
                        m[p] ^= 1 << (next() % 8);
                    }
                    1 => {
                        let p = (next() % m.len() as u64) as usize;
                        m[p] = next() as u8;
                    }
                    2 => {
                        let cut = (next() % m.len() as u64).max(1) as usize;
                        m.truncate(cut);
                    }
                    3 => {
                        let p = (next() % m.len() as u64) as usize;
                        m.insert(p, next() as u8);
                    }
                    4 => {
                        if m.len() > 1 {
                            let p = (next() % (m.len() as u64 - 1)) as usize;
                            m.remove(p);
                        }
                    }
                    _ => {
                        let n = 1 + (next() % 16) as usize;
                        m.extend(vec![next() as u8; n]);
                    }
                }
            }
            must_not_panic(&format!("stress-{cases}"), &m);
            if cases % 4 == 0 {
                must_not_panic_stream(&format!("stress-stream-{cases}"), &m);
            }
            cases += 1;
        }
    }
    println!("{cases} stress mutations survived without panic");
}
