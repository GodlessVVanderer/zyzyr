#!/bin/sh
set -e

usage() {
    echo "Usage: ./build-dropin.sh [OPTION]"
    echo ""
    echo "Build ZyZyR as a drop-in libzstd replacement."
    echo ""
    echo "Options:"
    echo "  --shared     Build shared library (.so / .dylib / .dll)  [default]"
    echo "  --static     Build static library (.a / .lib)"
    echo "  --both       Build shared and static"
    echo "  --help       Show this help"
    echo ""
    echo "Output goes to target/release/ with the libzstd name."
}

build_shared() {
    echo "Building shared library..."
    cargo rustc --release --features compat --crate-type cdylib
    case "$(uname -s)" in
        Linux*)
            cp target/release/libzyzyr.so target/release/libzstd.so
            echo "  -> target/release/libzstd.so"
            ;;
        Darwin*)
            cp target/release/libzyzyr.dylib target/release/libzstd.dylib
            echo "  -> target/release/libzstd.dylib"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            cp target/release/zyzyr.dll target/release/zstd.dll
            echo "  -> target/release/zstd.dll"
            ;;
    esac
}

build_static() {
    echo "Building static library..."
    cargo rustc --release --features compat --crate-type staticlib
    case "$(uname -s)" in
        Linux*|Darwin*)
            cp target/release/libzyzyr.a target/release/libzstd.a
            echo "  -> target/release/libzstd.a"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            cp target/release/zyzyr.lib target/release/zstd.lib
            echo "  -> target/release/zstd.lib"
            ;;
    esac
}

MODE="${1:---shared}"

case "$MODE" in
    --shared)  build_shared ;;
    --static)  build_static ;;
    --both)    build_shared; build_static ;;
    --help|-h) usage ;;
    *)         echo "Unknown option: $MODE"; usage; exit 1 ;;
esac

echo ""
echo "Done. Drop the output file where your program looks for libzstd."
echo "Header: include/zyzyr.h (or use existing zstd.h — the symbols match)."
