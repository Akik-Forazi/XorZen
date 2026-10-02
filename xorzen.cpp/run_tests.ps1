#!/usr/bin/env pwsh
<#
.SYNOPSIS
    XORZEN.CPP Integration Test Runner
    
.DESCRIPTION
    Comprehensive test suite for XORZEN.CPP components
    
.PARAMETER Verbose
    Enable verbose output
    
.PARAMETER Benchmark
    Run performance benchmarks
    
.PARAMETER Quick
    Run only critical tests (skip benchmarks)
    
.EXAMPLE
    .\run_tests.ps1 -Verbose -Benchmark
    .\run_tests.ps1 -Quick
#>

param(
    [switch]$Verbose,
    [switch]$Benchmark,
    [switch]$Quick
)

# Colors
$Green = [System.ConsoleColor]::Green
$Red = [System.ConsoleColor]::Red
$Yellow = [System.ConsoleColor]::Yellow
$Cyan = [System.ConsoleColor]::Cyan

Write-Host "========================================" -ForegroundColor $Cyan
Write-Host "XORZEN.CPP Integration Test Runner" -ForegroundColor $Cyan
Write-Host "========================================" -ForegroundColor $Cyan
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Check Python
Write-Host "Checking Python..." -ForegroundColor $Cyan
$pythonExe = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonExe) {
    Write-Host "ERROR: Python not found" -ForegroundColor $Red
    exit 1
}

$pythonVersion = & python --version 2>&1
Write-Host "Found: $pythonVersion" -ForegroundColor $Green

# Check PyTorch
Write-Host ""
Write-Host "Checking PyTorch..." -ForegroundColor $Cyan
$torchCheck = & python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Found: $torchCheck" -ForegroundColor $Green
} else {
    Write-Host "ERROR: PyTorch not installed" -ForegroundColor $Red
    exit 1
}

# Build command
$cmd = "python tests/test_integration.py"

if ($Verbose) {
    $cmd += " --verbose"
    Write-Host "Verbose mode: ON" -ForegroundColor $Yellow
}

if ($Benchmark -and -not $Quick) {
    $cmd += " --benchmark"
    Write-Host "Benchmarks: ENABLED" -ForegroundColor $Yellow
}

if ($Quick) {
    Write-Host "Mode: Quick (benchmarks disabled)" -ForegroundColor $Yellow
}

Write-Host ""
Write-Host "Running tests..." -ForegroundColor $Cyan
Write-Host "Command: $cmd" -ForegroundColor $Cyan
Write-Host ""

# Run tests
Push-Location $projectRoot
& python tests/test_integration.py $(if ($Verbose) { "--verbose" }) $(if ($Benchmark -and -not $Quick) { "--benchmark" })
$exitCode = $LASTEXITCODE
Pop-Location

# Summary
Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "========================================" -ForegroundColor $Green
    Write-Host "ALL TESTS PASSED" -ForegroundColor $Green
    Write-Host "========================================" -ForegroundColor $Green
} else {
    Write-Host "========================================" -ForegroundColor $Red
    Write-Host "TESTS FAILED" -ForegroundColor $Red
    Write-Host "========================================" -ForegroundColor $Red
}

exit $exitCode
