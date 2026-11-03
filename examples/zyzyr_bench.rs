use std::time::Instant;

#[cfg(windows)]
mod cycles {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> isize;
        fn QueryThreadCycleTime(thread: isize, cycles: *mut u64) -> i32;
    }

    pub fn now() -> Option<u64> {
        let mut c = 0u64;
        let ok = unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut c) };
        (ok != 0).then_some(c)
    }
}

#[cfg(not(windows))]
mod cycles {
    pub fn now() -> Option<u64> {
        None
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

#[inline(never)]
fn bench(path: &str, runs: usize, dict: Option<&zyzyr::Dictionary>) {
    let data = std::fs::read(path).expect("read input");
    let mut dec = zyzyr::Decoder::new();
    let mut out = Vec::new();
    let mut decode = |out: &mut Vec<u8>| {
        out.clear();
        let r = match dict {
            Some(d) => dec.decompress_into_with_dict(&data, d, out),
            None => dec.decompress_into(&data, out),
        };
        r.unwrap_or_else(|e| panic!("{path}: {e}"))
    };
    let n = decode(&mut out);
    let mut times = Vec::with_capacity(runs);
    let mut cyc = Vec::with_capacity(runs);
    for _ in 0..runs {
        let c0 = cycles::now();
        let t = Instant::now();
        let m = decode(&mut out);
        times.push(t.elapsed().as_secs_f64());
        if let (Some(a), Some(b)) = (c0, cycles::now()) {
            cyc.push((b - a) as f64 / n.max(1) as f64);
        }
        assert_eq!(m, n);
    }
    let med = median(&mut times);
    let mbs = |s: f64| n as f64 / s / 1e6;
    let cpb = if cyc.is_empty() {
        String::new()
    } else {
        format!("  cycles/B {:.9}", median(&mut cyc))
    };
    println!(
        "{path}  in {} B  out {n} B  runs {runs}  median {:.9} s  median {:.9} MB/s  best {:.9} MB/s  worst {:.9} MB/s{cpb}",
        data.len(),
        med,
        mbs(med),
        mbs(times[0]),
        mbs(times[runs - 1]),
    );
}

/// Runs `f` `depth` frames further down the stack. The decoder and its caller's frame move with it:
/// the same build measured at different depths differs by up to 0.85%, so a race of two builds runs
/// both at each of a set of depths.
#[inline(never)]
fn deeper(depth: usize, f: &mut dyn FnMut()) {
    let pad = std::hint::black_box([0u8; 40]);
    if depth == 0 {
        // Opaque, so `bench` is not inlined here and each level stays one small frame.
        std::hint::black_box(f)();
    } else {
        deeper(depth - 1, f);
    }
    std::hint::black_box(&pad);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = || -> ! {
        eprintln!("usage: zyzyr_bench <runs> [--dict <dictionary>] [--depth <frames>] <file.zst>...");
        std::process::exit(2)
    };
    let runs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
    let runs = runs.max(1);
    let mut dict = None;
    let mut depth = 0;
    let mut files = Vec::new();
    while let Some(a) = args.next() {
        if a == "--dict" {
            let p = args.next().unwrap_or_else(|| usage());
            let bytes = std::fs::read(&p).expect("read dictionary");
            dict = Some(zyzyr::Dictionary::new(&bytes).expect("dictionary"));
        } else if a == "--depth" {
            depth = args.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| usage());
        } else {
            files.push(a);
        }
    }
    if files.is_empty() {
        usage();
    }
    for f in &files {
        deeper(depth, &mut || bench(f, runs, dict.as_ref()));
    }
}
