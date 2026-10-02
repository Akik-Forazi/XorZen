# XORZEN.CPP GGML Setup Script
# Day 1: Copy GGML files and setup structure
# Created: May 26, 2026

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "XORZEN.CPP GGML INTEGRATION - DAY 1" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$LLAMA_CPP = "C:\Users\akikf\programing\llama.cpp"
$XORZEN_CPP = "C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen.cpp"
$GGML_SRC = "$LLAMA_CPP\ggml\src"
$GGML_INC = "$LLAMA_CPP\ggml\include"
$GGML_DEST = "$XORZEN_CPP\extern\ggml"

# Step 1: Copy GGML Core Files
Write-Host "[1/5] Copying GGML core files..." -ForegroundColor Yellow

$core_files = @(
    "ggml.c",
    "ggml.cpp",
    "ggml-impl.h",
    "ggml-common.h",
    "ggml-quants.c",
    "ggml-quants.h",
    "ggml-alloc.c",
    "ggml-backend.cpp",
    "ggml-backend-impl.h",
    "ggml-threading.cpp",
    "ggml-threading.h"
)

foreach ($file in $core_files) {
    $src = "$GGML_SRC\$file"
    $dst = "$GGML_DEST\$file"
    
    if (Test-Path $src) {
        Copy-Item $src $dst -Force
        Write-Host "  Copied $file" -ForegroundColor Green
    } else {
        Write-Host "  Not found: $file" -ForegroundColor Red
    }
}

# Step 2: Copy GGML Headers
Write-Host "`n[2/5] Copying GGML headers..." -ForegroundColor Yellow

if (Test-Path "$GGML_INC\ggml.h") {
    Copy-Item "$GGML_INC\ggml.h" "$GGML_DEST\ggml.h" -Force
    Write-Host "  Copied ggml.h" -ForegroundColor Green
}

if (Test-Path "$GGML_INC\ggml-alloc.h") {
    Copy-Item "$GGML_INC\ggml-alloc.h" "$GGML_DEST\ggml-alloc.h" -Force
    Write-Host "  Copied ggml-alloc.h" -ForegroundColor Green
}

if (Test-Path "$GGML_INC\ggml-backend.h") {
    Copy-Item "$GGML_INC\ggml-backend.h" "$GGML_DEST\ggml-backend.h" -Force
    Write-Host "  Copied ggml-backend.h" -ForegroundColor Green
}

# Step 3: Copy CPU-specific files
Write-Host "`n[3/5] Copying CPU optimizations..." -ForegroundColor Yellow

$cpu_src = "$GGML_SRC\ggml-cpu"
if (Test-Path "$cpu_src\ggml-cpu.c") {
    Copy-Item "$cpu_src\ggml-cpu.c" "$GGML_DEST\ggml-cpu.c" -Force
    Write-Host "  Copied ggml-cpu.c" -ForegroundColor Green
}
if (Test-Path "$cpu_src\ggml-cpu.h") {
    Copy-Item "$cpu_src\ggml-cpu.h" "$GGML_DEST\ggml-cpu.h" -Force
    Write-Host "  Copied ggml-cpu.h" -ForegroundColor Green
}

# Step 4: Copy GGUF (model format)
Write-Host "`n[4/5] Copying GGUF format..." -ForegroundColor Yellow
if (Test-Path "$GGML_SRC\gguf.cpp") {
    Copy-Item "$GGML_SRC\gguf.cpp" "$GGML_DEST\gguf.cpp" -Force
    Write-Host "  Copied gguf.cpp" -ForegroundColor Green
}
if (Test-Path "$GGML_INC\gguf.h") {
    Copy-Item "$GGML_INC\gguf.h" "$GGML_DEST\gguf.h" -Force
    Write-Host "  Copied gguf.h" -ForegroundColor Green
}

# Step 5: Verify
Write-Host "`n[5/5] Verification..." -ForegroundColor Yellow

$required = @("ggml.c", "ggml.h", "ggml-quants.c", "ggml-quants.h")
$all_ok = $true

foreach ($file in $required) {
    if (Test-Path "$GGML_DEST\$file") {
        Write-Host "  $file exists" -ForegroundColor Green
    } else {
        Write-Host "  $file missing!" -ForegroundColor Red
        $all_ok = $false
    }
}

Write-Host ""
if ($all_ok) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "SUCCESS! GGML files copied successfully" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Review CMakeLists.txt changes (will be generated)" -ForegroundColor White
    Write-Host "2. Build with: cd build-msvc; cmake --build . -j8" -ForegroundColor White
    Write-Host "3. Test ggml_bridge" -ForegroundColor White
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "ERROR: Some files are missing!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
}

Write-Host ""
