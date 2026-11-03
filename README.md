# ZyZyR

Zstandard (RFC 8878) decompressor in pure register-mapped Rust. No assembly, no dependencies, `no_std` capable.

## Performance

Measured on Ryzen 7 4800HS (Zen 2, AVX2), median of 5 runs. Both decompressors read from the same zstd-compressed buffer.

| Data | Size | Level | ZyZyR MB/s | libzstd MB/s | Speedup |
|------|------|-------|-----------|-------------|---------|
| text | 1 MB | 1 | 10,766 | 2,901 | **3.7x** |
| text | 1 MB | 19 | 11,348 | 3,380 | **3.4x** |
| binary | 1 MB | 1 | 45,197 | 3,705 | **12.2x** |
| binary | 1 MB | 19 | 39,719 | 3,779 | **10.5x** |
| text | 16 MB | 1 | 4,164 | 1,730 | **2.4x** |
| text | 64 MB | 19 | 7,292 | 3,602 | **2.0x** |
| text | 256 MB | 1 | 6,872 | 3,692 | **1.9x** |
| text | 256 MB | 19 | 7,922 | 3,861 | **2.1x** |
| binary | 256 MB | 19 | 6,959 | 3,107 | **2.2x** |

Small incompressible data shows the highest ratio because the fast path for stored blocks is essentially a bounds-checked memcpy. At 256 MB where cache misses dominate, ZyZyR holds 1.9-2.2x.

Run the benchmark yourself:

```
cargo run --release --example vs_libzstd -- 5
```

## Usage

```rust
let compressed: &[u8] = /* zstd frame */;
let mut decoder = zyzyr::Decoder::new();
let mut output = Vec::new();
decoder.decompress_into(compressed, &mut output).unwrap();
```

For `no_std` environments, disable the default `std` feature:

```toml
[dependencies]
zyzyr = { version = "0.1", default-features = false }
```

## Drop-in libzstd replacement

The `compat` feature provides a complete libzstd-compatible C API. Same function names, same types, same error codes:

- `ZSTD_decompress`, `ZSTD_decompressDCtx`
- `ZSTD_createDCtx` / `ZSTD_freeDCtx`
- `ZSTD_decompressStream` with `ZSTD_inBuffer` / `ZSTD_outBuffer`
- `ZSTD_getFrameContentSize`, `ZSTD_findFrameCompressedSize`
- `ZSTD_isError`, `ZSTD_getErrorName`
- Dictionary support, streaming, parameter setting, context reset

Build as a shared library:

```
cargo rustc --release --features compat --crate-type cdylib
```

Name the output `libzstd.so` / `zstd.dll` / `libzstd.dylib`. Any program linked against libzstd for decompression picks up ZyZyR instead. No code changes needed.

ZyZyR also has its own C API (`zyzyr_*` symbols, header at `include/zyzyr.h`) with the `capi` feature.

## How it works

Every hot loop maps its working state to named registers and lanes so the compiler emits the same machine code that hand-written intrinsics would. The Huffman decoder, FSE decoder, and sequence executor are all written this way. No lookup tables where register work is faster, no abstraction layers between the algorithm and the machine.

## What it covers

Full RFC 8878: frames, blocks, literals (raw / RLE / compressed / treeless), sequences with match/literal/offset coding, Huffman and FSE entropy decoding, dictionaries, content checksums (xxHash64), skippable frames, multi-frame concatenation.

## License

AGPL-3.0-only. Commercial licenses available for use in proprietary software.
