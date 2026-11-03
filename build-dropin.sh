#!/bin/sh
set -e

cargo rustc --release --features compat --crate-type cdylib

case "$(uname -s)" in
    Linux*)   cp target/release/libzyzyr.so  target/release/libzstd.so   ;;
    Darwin*)  cp target/release/libzyzyr.dylib target/release/libzstd.dylib ;;
    MINGW*|MSYS*|CYGWIN*) cp target/release/zyzyr.dll target/release/zstd.dll ;;
esac

echo "Built drop-in replacement in target/release/"
ls -lh target/release/libzstd.* target/release/zstd.* 2>/dev/null || true
