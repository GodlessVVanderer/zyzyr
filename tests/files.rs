mod common;

use std::fs;
use std::panic;
use std::path::{Path, PathBuf};

use common::{cases, corpus, decode, not_run};

#[test]
fn in_crate_frames_decode_to_their_recorded_outputs() {
    let mut dec = zyzyr::Decoder::new();
    for c in cases() {
        let out = decode(&c).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        assert!(c.check(&out), "{}: {} bytes, sha-256 differs", c.name, out.len());
        let mut again = b"keep".to_vec();
        let n = match c.dictionary() {
            Some(d) => dec.decompress_into_with_dict(&c.data, &d, &mut again),
            None => dec.decompress_into(&c.data, &mut again),
        }
        .unwrap();
        assert_eq!(n, c.len, "{}", c.name);
        assert!(again.starts_with(b"keep") && c.check(&again[4..]), "{}", c.name);
        println!("{}: {} -> {} bytes, sha-256 matches", c.name, c.data.len(), out.len());
    }
}

#[test]
fn dictionary_frames_need_their_dictionary() {
    let mut n = 0;
    for c in cases().into_iter().filter(|c| c.dict.is_some()) {
        assert!(zyzyr::decompress(&c.data).is_err(), "{}", c.name);
        let d = c.dictionary().unwrap();
        assert_eq!(zyzyr::dictionary_id(&c.data).unwrap(), d.id(), "{}", c.name);
        n += 1;
    }
    assert!(n >= 2);
}

fn no_panic(name: &str, what: &str, data: &[u8]) {
    let r = panic::catch_unwind(|| zyzyr::decompress(data));
    assert!(r.is_ok(), "{name}: panic on {what}");
}

#[test]
fn malformed_inputs_never_panic() {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut cases_run = 0usize;
    for c in cases() {
        let data = &c.data;
        let step = (data.len() / 512).max(1);
        let mut k = 0;
        while k < data.len() {
            no_panic(&c.name, &format!("prefix {k}"), &data[..k]);
            cases_run += 1;
            k += step;
        }
        let rounds = if data.len() < 4096 { 800 } else { 300 };
        for i in 0..rounds {
            let mut m = data.clone();
            for _ in 0..1 + next() % 4 {
                let p = (next() % m.len() as u64) as usize;
                m[p] = match next() % 3 {
                    0 => m[p] ^ (1 << (next() % 8)),
                    1 => next() as u8,
                    _ => m[p].wrapping_add(1),
                };
            }
            if next() % 4 == 0 {
                let cut = (next() % m.len() as u64).max(1) as usize;
                m.truncate(cut);
            }
            no_panic(&c.name, &format!("mutation {i}"), &m);
            cases_run += 1;
        }
    }
    println!("{cases_run} malformed cases, no panic");
}

fn entries(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
        Err(_) => Vec::new(),
    };
    v.sort();
    v
}

fn zst_files(dir: &Path) -> Vec<PathBuf> {
    entries(dir).into_iter().filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "zst")).collect()
}

fn original_for(z: &Path) -> Option<Vec<u8>> {
    let mut stem = z.file_name()?.to_str()?.strip_suffix(".zst")?;
    let dir = z.parent()?;
    loop {
        let p = dir.join(stem);
        if p.is_file() {
            return fs::read(p).ok();
        }
        stem = &stem[..stem.rfind('.')?];
    }
}

fn dictionaries(dir: &Path) -> Vec<zyzyr::Dictionary> {
    entries(dir)
        .into_iter()
        .filter(|p| p.is_file() && p.extension().is_none_or(|e| e != "zst"))
        .filter_map(|p| fs::read(p).ok())
        .filter(|b| b.len() >= 8 && b[..4] == [0x37, 0xA4, 0x30, 0xEC])
        .filter_map(|b| zyzyr::Dictionary::new(&b).ok())
        .collect()
}

#[test]
fn external_corpus() {
    let Some(root) = corpus() else {
        not_run("external corpus cross-check");
        return;
    };
    let (mut files, mut originals, mut references) = (0, 0, 0);
    let mut dirs = vec![root.clone()];
    dirs.extend(entries(&root).into_iter().filter(|p| p.is_dir() && p.file_name().is_some_and(|n| n != "refd")));
    for dir in dirs {
        let dicts = dictionaries(&dir);
        let group = if dir == root { "scratch".to_string() } else { dir.file_name().unwrap().to_string_lossy().into_owned() };
        for z in zst_files(&dir) {
            let data = fs::read(&z).unwrap();
            let out = match zyzyr::decompress(&data) {
                Err(zyzyr::Error::Dictionary(id)) => match dicts.iter().find(|d| d.id() == id) {
                    Some(d) => zyzyr::decompress_with_dict(&data, d),
                    None => continue,
                },
                r => r,
            };
            let Ok(out) = out else {
                continue;
            };
            if let Some(o) = original_for(&z) {
                assert!(out == o, "{} differs from its original", z.display());
                originals += 1;
            }
            let stem = z.file_name().unwrap().to_string_lossy().trim_end_matches(".zst").to_string();
            if let Ok(r) = fs::read(root.join("refd").join(&group).join(stem)) {
                assert!(out == r, "{} differs from the reference decoder output", z.display());
                references += 1;
            }
            files += 1;
        }
    }
    println!("external corpus: {files} frames decoded, {references} byte-identical to reference decodes, {originals} to their originals");
    assert!(files > 0);
}
