pub const MANIFEST: &str = include_str!("manifest.txt");

pub const FILES: &[(&str, &[u8])] = &[
    ("text.l1.zst", include_bytes!("text.l1.zst")),
    ("text64.l19.zst", include_bytes!("text64.l19.zst")),
    ("text64.l3-nc.zst", include_bytes!("text64.l3-nc.zst")),
    ("text64.w10.zst", include_bytes!("text64.w10.zst")),
    ("text64.nocs.zst", include_bytes!("text64.nocs.zst")),
    ("small.zst", include_bytes!("small.zst")),
    ("rand.zst", include_bytes!("rand.zst")),
    ("runs.zst", include_bytes!("runs.zst")),
    ("records.l1.zst", include_bytes!("records.l1.zst")),
    ("records.l19.zst", include_bytes!("records.l19.zst")),
    ("same.zst", include_bytes!("same.zst")),
    ("multi.zst", include_bytes!("multi.zst")),
    ("skip.zst", include_bytes!("skip.zst")),
    ("rle.zst", include_bytes!("rle.zst")),
    ("dicted.zst", include_bytes!("dicted.zst")),
    ("dicted.l19.zst", include_bytes!("dicted.l19.zst")),
    ("patch.zst", include_bytes!("patch.zst")),
    ("samples.dict", include_bytes!("samples.dict")),
    ("base.txt", include_bytes!("base.txt")),
];

pub fn file(name: &str) -> &'static [u8] {
    FILES.iter().find(|(n, _)| *n == name).map(|(_, d)| *d).unwrap()
}
