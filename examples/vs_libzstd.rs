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
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn mbs(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / secs / 1e6
}

struct Timing {
    label: &'static str,
    decoded_bytes: usize,
    times: Vec<f64>,
    cyc: Vec<f64>,
}

impl Timing {
    fn new(label: &'static str) -> Self {
        Timing { label, decoded_bytes: 0, times: Vec::new(), cyc: Vec::new() }
    }

    fn report(&mut self) {
        let n = self.decoded_bytes;
        let med = median(&mut self.times);
        let cpb = if self.cyc.is_empty() {
            String::new()
        } else {
            format!("  cycles/B {:.3}", median(&mut self.cyc))
        };
        println!(
            "{:<12} out {} B  median {:.9} s  {:.3} MB/s  best {:.3} MB/s  worst {:.3} MB/s{cpb}",
            self.label,
            n,
            med,
            mbs(n, med),
            mbs(n, self.times[0]),
            mbs(n, *self.times.last().unwrap()),
        );
    }
}

fn bench_zyzyr(compressed: &[u8], runs: usize) -> Timing {
    let mut t = Timing::new("zyzyr");
    let mut dec = zyzyr::Decoder::new();
    let mut out = Vec::new();
    dec.decompress_into(compressed, &mut out).expect("zyzyr warmup");
    t.decoded_bytes = out.len();
    for _ in 0..runs {
        out.clear();
        let c0 = cycles::now();
        let start = Instant::now();
        let n = dec.decompress_into(compressed, &mut out).expect("zyzyr");
        let elapsed = start.elapsed().as_secs_f64();
        t.times.push(elapsed);
        if let (Some(a), Some(b)) = (c0, cycles::now()) {
            t.cyc.push((b - a) as f64 / n.max(1) as f64);
        }
    }
    t
}

fn bench_libzstd(compressed: &[u8], runs: usize) -> Timing {
    let mut t = Timing::new("libzstd");
    let out = zstd::bulk::decompress(compressed, 512 * 1024 * 1024).expect("libzstd warmup");
    t.decoded_bytes = out.len();
    let cap = out.len() + 1024;
    for _ in 0..runs {
        let c0 = cycles::now();
        let start = Instant::now();
        let decoded = zstd::bulk::decompress(compressed, cap).expect("libzstd");
        let elapsed = start.elapsed().as_secs_f64();
        let n = decoded.len();
        t.times.push(elapsed);
        if let (Some(a), Some(b)) = (c0, cycles::now()) {
            t.cyc.push((b - a) as f64 / n.max(1) as f64);
        }
        std::hint::black_box(&decoded);
    }
    t
}

fn generate_text(size: usize) -> Vec<u8> {
    let seed: &[u8] = b"The quick brown fox jumps over the lazy dog. \
        Pack my box with five dozen liquor jugs. \
        How vexingly quick daft zebras jump. ";
    let mut buf = Vec::with_capacity(size);
    let mut i = 0u64;
    while buf.len() < size {
        let chunk = &seed[..(seed.len()).min(size - buf.len())];
        buf.extend_from_slice(chunk);
        i = i.wrapping_mul(6364136223846793005).wrapping_add(1);
        if buf.len() + 8 <= size {
            buf.extend_from_slice(&i.to_le_bytes());
        }
    }
    buf.truncate(size);
    buf
}

fn generate_binary(size: usize) -> Vec<u8> {
    let mut buf = vec![0u8; size];
    let mut s = 0x12345678u64;
    for chunk in buf.chunks_mut(8) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let b = s.to_le_bytes();
        for (d, &v) in chunk.iter_mut().zip(b.iter()) {
            *d = v;
        }
    }
    buf
}

fn main() {
    let mut args = std::env::args().skip(1);
    let runs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(50);
    let runs = runs.max(5);

    let levels = [1, 3, 19];
    let sizes_mb = [1, 4, 16, 64, 256];

    println!("=== zyzyr vs libzstd decompression ===");
    println!("runs per measurement: {runs}");
    println!();

    for &size_mb in &sizes_mb {
        let size = size_mb * 1024 * 1024;

        for (name, data) in [("text", generate_text(size)), ("binary", generate_binary(size))] {
            for &level in &levels {
                let compressed = zstd::bulk::compress(&data, level).expect("compress");
                let ratio = data.len() as f64 / compressed.len() as f64;

                println!("--- {name} {size_mb} MB  level {level}  compressed {} B  ratio {ratio:.2}x ---", compressed.len());

                let mut zy = bench_zyzyr(&compressed, runs);
                let mut lz = bench_libzstd(&compressed, runs);
                zy.report();
                lz.report();

                let zy_med = median(&mut zy.times.clone());
                let lz_med = median(&mut lz.times.clone());
                let speedup = lz_med / zy_med;
                if speedup >= 1.0 {
                    println!("  zyzyr {:.2}x faster", speedup);
                } else {
                    println!("  libzstd {:.2}x faster", 1.0 / speedup);
                }
                println!();
            }
        }
    }

    let file_args: Vec<String> = args.collect();
    for path in &file_args {
        let compressed = std::fs::read(path).expect("read file");
        println!("--- file: {path}  {} B compressed ---", compressed.len());
        let mut zy = bench_zyzyr(&compressed, runs);
        let mut lz = bench_libzstd(&compressed, runs);
        zy.report();
        lz.report();
        let zy_med = median(&mut zy.times.clone());
        let lz_med = median(&mut lz.times.clone());
        let speedup = lz_med / zy_med;
        if speedup >= 1.0 {
            println!("  zyzyr {:.2}x faster", speedup);
        } else {
            println!("  libzstd {:.2}x faster", 1.0 / speedup);
        }
        println!();
    }
}
