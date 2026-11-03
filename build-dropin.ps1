$ErrorActionPreference = 'Stop'

cargo rustc --release --features compat --crate-type cdylib

Copy-Item target\release\zyzyr.dll target\release\zstd.dll -Force
Write-Host "Built drop-in: target\release\zstd.dll"
Get-Item target\release\zstd.dll | Select-Object Name, Length
