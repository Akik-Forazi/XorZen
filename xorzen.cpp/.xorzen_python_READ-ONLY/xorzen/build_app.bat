@echo off
echo ============================================================
echo   XORZEN APP BUILDER - NUITKA COMPILATION
echo   FRAZIYM AI (c) 2026
echo ============================================================

REM 1. Install Nuitka if missing
pip install nuitka zstandard

REM 2. Compile XORZEN into a standalone Windows executable
REM --standalone: bundle dependencies
REM --onefile: compress into single .exe
REM --windows-disable-console: no terminal window
REM --enable-plugin=tk-inter: support for dashboard
REM --include-data-dir: bundle tokenizers and metadata
REM --include-package-data: include compiled .pyd kernels

python -m nuitka ^
    --standalone ^
    --onefile ^
    --windows-disable-console ^
    --enable-plugin=tk-inter ^
    --output-dir=dist ^
    --output-filename=XorZen_Neural_Engine.exe ^
    --include-data-dir=tokenizer/pretrained=xorzen/tokenizer/pretrained ^
    --include-package-data=xorzen.speed ^
    --company-name="FRAZIYM TECH" ^
    --product-name="XorZen Neural Engine" ^
    --file-version=0.2.4 ^
    --product-version=0.2.4 ^
    --copyright="Copyright (c) 2026 Akik Faraji, FRAZIYM AI" ^
    xorzen_app.py

echo.
echo ============================================================
echo   BUILD COMPLETE: dist/XorZen_Neural_Engine.exe
echo ============================================================
pause
