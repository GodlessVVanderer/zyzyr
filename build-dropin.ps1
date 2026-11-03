param(
    [ValidateSet('shared', 'static', 'both', 'help')]
    [string]$Mode = 'shared'
)

$ErrorActionPreference = 'Stop'

function Show-Help {
    Write-Host @"
Usage: .\build-dropin.ps1 [-Mode shared|static|both|help]

Build ZyZyR as a drop-in libzstd replacement.

Modes:
  shared    Build zstd.dll (default)
  static    Build zstd.lib
  both      Build both
  help      Show this help

Output goes to target\release\ with the libzstd name.
"@
}

function Build-Shared {
    Write-Host "Building shared library..."
    cargo rustc --release --features compat --crate-type cdylib
    Copy-Item target\release\zyzyr.dll target\release\zstd.dll -Force
    Write-Host "  -> target\release\zstd.dll"
    Get-Item target\release\zstd.dll | Select-Object Name, @{N='Size';E={'{0:N0} KB' -f ($_.Length/1KB)}}
}

function Build-Static {
    Write-Host "Building static library..."
    cargo rustc --release --features compat --crate-type staticlib
    Copy-Item target\release\zyzyr.lib target\release\zstd.lib -Force
    Write-Host "  -> target\release\zstd.lib"
    Get-Item target\release\zstd.lib | Select-Object Name, @{N='Size';E={'{0:N0} KB' -f ($_.Length/1KB)}}
}

switch ($Mode) {
    'shared' { Build-Shared }
    'static' { Build-Static }
    'both'   { Build-Shared; Build-Static }
    'help'   { Show-Help }
}

Write-Host ""
Write-Host "Done. Drop the output where your program looks for zstd.dll."
Write-Host "Header: include\zyzyr.h (or use existing zstd.h - the symbols match)."
