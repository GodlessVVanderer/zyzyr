mod common;

use std::fs;
use std::path::Path;

use common::{case, cases, corpus, not_run, Case};
use zyzyr::{Dictionary, Error, StreamDecoder};

const SLACK: usize = 32;
const BLOCK: usize = 128 * 1024;

fn window_bound(z: &[u8]) -> usize {
    let mut p = 0usize;
    while p + 8 <= z.len() && u32::from_le_bytes([z[p], z[p + 1], z[p + 2], z[p + 3]]) & 0xFFFF_FFF0 == 0x184D_2A50 {
        p += 8 + u32::from_le_bytes([z[p + 4], z[p + 5], z[p + 6], z[p + 7]]) as usize;
    }
    if p + 6 > z.len() {
        return 0;
    }
    let content = zyzyr::content_size(z).ok().flatten().map_or(usize::MAX, |n| n as usize);
    let fhd = z[p + 4];
    if fhd & 0x20 != 0 {
        return content;
    }
    let wd = z[p + 5];
    let base = 1usize << (10 + (wd >> 3));
    (base + (base >> 3) * (wd & 7) as usize).min(content)
}

fn stream(z: &[u8], chunk: usize, out_cap: usize, dict: Option<&Dictionary>) -> (Result<(), Error>, Vec<u8>, usize) {
    let mut d = StreamDecoder::new();
    let mut got = Vec::new();
    let mut buf = vec![0u8; out_cap];
    let mut peak = 0usize;
    for c in z.chunks(chunk.max(1)) {
        let mut rest = c;
        loop {
            let r = match dict {
                Some(k) => d.feed_with_dict(rest, k),
                None => d.feed(rest),
            };
            let n = match r {
                Ok(n) => n,
                Err(e) => return (Err(e), got, peak),
            };
            rest = &rest[n..];
            peak = peak.max(d.memory());
            loop {
                let k = d.drain(&mut buf);
                if k == 0 {
                    break;
                }
                got.extend_from_slice(&buf[..k]);
            }
            if rest.is_empty() {
                break;
            }
        }
    }
    (d.finish(), got, peak)
}

fn check_case(c: &Case) -> usize {
    let dict = c.dictionary();
    let chunkings: &[(usize, usize)] = if c.data.len() > 30_000 {
        &[(7, 333), (4096, 100), (131_075, 131_072), (usize::MAX, 64)]
    } else {
        &[(1, 7), (13, 1), (4096, 100), (usize::MAX, 64)]
    };
    let bound = window_bound(&c.data).saturating_add(3 * (BLOCK + SLACK));
    let mut most = 0;
    for &(ic, oc) in chunkings {
        let (r, got, peak) = stream(&c.data, ic.min(c.data.len().max(1)), oc, dict.as_ref());
        assert_eq!(r, Ok(()), "{} in {ic} out {oc}", c.name);
        assert!(c.check(&got), "{} in {ic} out {oc}: {} bytes", c.name, got.len());
        assert!(peak <= bound, "{} in {ic}: {peak} bytes held, bound {bound}", c.name);
        most = most.max(peak);
    }
    most
}

#[test]
fn in_crate_frames_stream_identically_within_the_memory_bound() {
    for c in cases() {
        let most = check_case(&c);
        println!("{}: {} bytes streamed identically, at most {most} bytes held", c.name, c.len);
    }
}

#[test]
fn small_window_frame_wraps_the_ring() {
    let c = case("text64.w10.zst");
    let (r, got, peak) = stream(&c.data, 3000, 4096, None);
    assert_eq!(r, Ok(()));
    assert!(c.check(&got));
    assert!(peak < 16 * 1024, "{peak}");
    println!("1 KiB window frame: {} bytes out, at most {peak} bytes held", got.len());
}

#[test]
fn truncation_and_corruption_are_reported() {
    let c = case("text64.l3-nc.zst");
    let (r, _, _) = stream(&c.data[..c.data.len() - 5], 4096, 4096, None);
    assert_eq!(r, Err(Error::Truncated));
    let c = case("text.l1.zst");
    let mut bad = c.data.clone();
    let k = bad.len() / 2;
    bad[k] ^= 0x55;
    let (r, _, _) = stream(&bad, 4096, 4096, None);
    assert!(r.is_err());
    let (r, _, _) = stream(b"not a frame at all", 5, 16, None);
    assert_eq!(r, Err(Error::BadMagic));
    let mut d = StreamDecoder::new();
    assert_eq!(d.feed(&[]), Ok(0));
    assert_eq!(d.finish(), Ok(()));
}

#[test]
fn skippable_frames_stream_in_pieces() {
    let c = case("text.l1.zst");
    let skip = |n: u32, tag: u8| {
        let mut f = (0x184D_2A50u32 | tag as u32).to_le_bytes().to_vec();
        f.extend_from_slice(&n.to_le_bytes());
        f.extend((0..n).map(|i| i as u8));
        f
    };
    let mut z = skip(5, 0);
    z.extend_from_slice(&c.data);
    z.extend(skip(0, 7));
    z.extend(skip(300, 15));
    z.extend_from_slice(&c.data);
    let want = zyzyr::decompress(&z).unwrap();
    for chunk in [1usize, 2, 3, 7, 64, 4096, z.len()] {
        let (r, got, _) = stream(&z, chunk, 100, None);
        assert_eq!(r, Ok(()), "chunk {chunk}");
        assert!(got == want, "chunk {chunk}");
    }
}

#[test]
fn every_descriptor_agrees_with_one_shot() {
    let body: Vec<u8> = (0..300u32).map(|i| (i * 7 + 3) as u8).collect();
    let mut n = 0;
    for fhd in 0..=255u8 {
        for did in [0u32, 0x0102_0304] {
            for size in [300u64, 5] {
                let mut z = 0xFD2F_B528u32.to_le_bytes().to_vec();
                z.push(fhd);
                let single = fhd & 0x20 != 0;
                if !single {
                    z.push(0);
                }
                let d = [0, 1, 2, 4][(fhd & 3) as usize];
                z.extend_from_slice(&did.to_le_bytes()[..d]);
                match fhd >> 6 {
                    0 if single => z.push(size as u8),
                    0 => {}
                    1 => z.extend_from_slice(&(size.wrapping_sub(256) as u16).to_le_bytes()),
                    2 => z.extend_from_slice(&(size as u32).to_le_bytes()),
                    _ => z.extend_from_slice(&size.to_le_bytes()),
                }
                z.extend_from_slice(&((body.len() as u32) << 3 | 1).to_le_bytes()[..3]);
                z.extend_from_slice(&body);
                if fhd & 4 != 0 {
                    z.extend_from_slice(&[1, 2, 3, 4]);
                }
                let want = zyzyr::decompress(&z);
                for chunk in [1, z.len()] {
                    let (r, got, _) = stream(&z, chunk, 100, None);
                    match &want {
                        Ok(w) => assert!(r == Ok(()) && got == *w, "descriptor {fhd:#04x} did {did} size {size} chunk {chunk}"),
                        Err(e) => assert_eq!(r, Err(*e), "descriptor {fhd:#04x} did {did} size {size} chunk {chunk}"),
                    }
                }
                n += 1;
            }
        }
    }
    println!("{n} frame headers: streaming and one-shot agree");
}

#[test]
fn mutated_inputs_agree_with_one_shot() {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut n = 0;
    for c in cases().into_iter().filter(|c| c.dict.is_none()) {
        for _ in 0..150 {
            let mut m = c.data.clone();
            for _ in 0..1 + next() % 3 {
                let p = (next() % m.len() as u64) as usize;
                m[p] ^= 1 << (next() % 8);
            }
            let a = zyzyr::decompress(&m);
            let (r, got, _) = stream(&m, 1 + (next() % 5000) as usize, 1 + (next() % 3000) as usize, None);
            match a {
                Ok(want) => {
                    assert_eq!(r, Ok(()));
                    assert!(got == want);
                }
                Err(_) => assert!(r.is_err()),
            }
            n += 1;
        }
    }
    println!("{n} mutated inputs: streaming and one-shot agree");
}

#[test]
fn external_corpus_streams_identically() {
    let Some(root) = corpus() else {
        not_run("external corpus streaming cross-check");
        return;
    };
    let mut n = 0;
    let Ok(rd) = fs::read_dir(&root) else {
        return;
    };
    for p in rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "zst")) {
        let data = fs::read(&p).unwrap();
        let Ok(want) = zyzyr::decompress(&data) else {
            continue;
        };
        let (r, got, _) = stream(&data, 65_536, 131_072, None);
        assert_eq!(r, Ok(()), "{}", Path::new(&p).display());
        assert!(got == want, "{}", p.display());
        n += 1;
    }
    println!("external corpus: {n} frames streamed identically");
}
