# 🚀 XORZEN.CPP - GET IT WORKING TODAY

**Goal**: Build the DLL and run smoke tests in the next 2 hours.

---

## ✅ OPTION 1: MSVC Build (RECOMMENDED - 90 mins)

### Step 1: Install Visual Studio 2022 Build Tools (60 mins)

1. **Download** Visual Studio Build Tools:
   ```
   https://visualstudio.microsoft.com/downloads/
   → Scroll to "All Downloads"
   → "Tools for Visual Studio"  
   → "Build Tools for Visual Studio 2022"
   ```

2. **Run installer** (`vs_BuildTools.exe`)

3. **Select workload**: "Desktop development with C++"
   - ✅ MSVC v143 - VS 2022 C++ x64/x86 build tools
   - ✅ Windows 10/11 SDK (latest)
   - ✅ C++ CMake tools for Windows
   - ✅ C++ ATL for latest build tools

4. **Install** (~5GB download + install)

### Step 2: Build with MSVC (15 mins)

1. **Open "Developer Command Prompt for VS 2022"**
   - Start Menu → Visual Studio 2022 → Developer Command Prompt
   - **NOT regular PowerShell!**

2. **Navigate to project**:
   ```cmd
   cd C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen.cpp
   ```

3. **Configure CMake**:
   ```cmd
   cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
     -DCMAKE_PREFIX_PATH="C:\Users\akikf\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\share\cmake"
   ```

4. **Build** (use all 8 cores):
   ```cmd
   cmake --build build --config Release -j8
   ```

### Step 3: Verify Build (5 mins)

```cmd
cd build\Release

:: Check DLL exists
dir XorZen_BETA-v-0.2.5_win64_backend.dll

:: Check executables
dir xorzen_train.exe
dir xorzen_infer.exe
```

### Step 4: Run Smoke Tests (10 mins)

```cmd
:: Test 1: Training smoke test
xorzen_train.exe

:: Test 2: Inference smoke test
xorzen_infer.exe
```

**Expected Output**: No crashes, some console logs

---

## ✅ OPTION 2: Pre-built LibTorch + MinGW (Faster but riskier)

### Step 1: Download LibTorch (20 mins)

1. Go to: https://pytorch.org/get-started/locally/
2. Select:
   - **PyTorch Build**: Stable
   - **Your OS**: Windows
   - **Package**: LibTorch
   - **Language**: C++/Java
   - **Compute Platform**: CPU
   - **Binaries**: cxx11 ABI

3. Download ZIP (~400MB)

4. Extract to `C:\libtorch`

### Step 2: Build with MinGW (10 mins)

```powershell
cd C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen.cpp

# Configure
cmake -S . -B build-mingw -G "MinGW Makefiles" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="C:\libtorch"

# Build
cd build-mingw
mingw32-make -j8
```

### Step 3: Verify & Test

```powershell
cd build-mingw

# Check outputs
dir XorZen_BETA-v-0.2.5_win64_backend.dll
dir xorzen_train.exe
dir xorzen_infer.exe

# Run tests
.\xorzen_train.exe
.\xorzen_infer.exe
```

---

## 🐛 TROUBLESHOOTING

### Issue: CMake can't find Torch

**Symptom**:
```
Could not find package configuration file provided by "Torch"
```

**Fix**:
```powershell
# Verify path exists
dir "C:\Users\akikf\AppData\Local\Programs\Python\Python313\Lib\site-packages\torch\share\cmake"

# If not, find correct path:
python -c "import torch; print(torch.utils.cmake_prefix_path)"
```

### Issue: MSVC flags error with MinGW

**Symptom**:
```
c++.exe: error: unrecognized command line option '/permissive-'
```

**Fix**: Use MSVC build (Option 1) OR download pre-built LibTorch (Option 2)

### Issue: Linker errors

**Symptom**:
```
undefined reference to `torch::*`
```

**Fix**: Ensure `${TORCH_LIBRARIES}` is linked in CMakeLists.txt (already done)

### Issue: DLL not found at runtime

**Symptom**:
```
xorzen_train.exe: The code execution cannot proceed because torch_cpu.dll was not found
```

**Fix**: CMake should auto-copy DLLs. If not:
```powershell
# Copy manually
copy C:\libtorch\lib\*.dll build\Release\
```

---

## ⚡ QUICK STATUS CHECK

After build completes, run this to verify everything worked:

```powershell
# Check all artifacts exist
$artifacts = @(
    "build\Release\XorZen_BETA-v-0.2.5_win64_backend.dll",
    "build\Release\xorzen_train.exe",
    "build\Release\xorzen_infer.exe"
)

foreach ($file in $artifacts) {
    if (Test-Path $file) {
        Write-Host "✅ $file" -ForegroundColor Green
    } else {
        Write-Host "❌ $file MISSING" -ForegroundColor Red
    }
}

# Get DLL size
$dll = Get-Item "build\Release\XorZen_BETA-v-0.2.5_win64_backend.dll"
Write-Host "DLL Size: $($dll.Length / 1MB) MB"

# Run quick test
Write-Host "`nRunning smoke test..."
cd build\Release
.\xorzen_train.exe
```

Expected output:
```
✅ build\Release\XorZen_BETA-v-0.2.5_win64_backend.dll
✅ build\Release\xorzen_train.exe
✅ build\Release\xorzen_infer.exe
DLL Size: ~15-20 MB

Running smoke test...
[XorzenModel] Initializing with config: vocab=10000, hidden=768, layers=12...
[Training] Step 1/100...
```

---

## 📝 WHAT TO DO AFTER SUCCESSFUL BUILD

### 1. Document the Build
```powershell
# Save build info
cmake --build build --target help > build_targets.txt
```

### 2. Run All Tests
```powershell
cd build\Release
.\xorzen_train.exe > train_output.log
.\xorzen_infer.exe > infer_output.log
```

### 3. Check Against Python
```powershell
# Compare with Python model
cd ..\..\..
python -m xorzen.xorzen_app
```

### 4. Report Back
Create a quick status file:
```powershell
@"
# Build Success Report
Date: $(Get-Date)
Build Tool: [MSVC or MinGW]
DLL Size: $((Get-Item build\Release\XorZen_BETA-v-0.2.5_win64_backend.dll).Length / 1MB) MB
Smoke Tests: [PASSED/FAILED]
Next Steps: [Unit tests / Optimization / Integration]
"@ | Out-File BUILD_SUCCESS.md
```

---

## 🎯 TIMELINE

| Task | Duration | Status |
|------|----------|--------|
| Install VS Build Tools | 60 mins | ⏳ TODO |
| Configure CMake | 5 mins | ⏳ TODO |
| Build DLL | 10 mins | ⏳ TODO |
| Run smoke tests | 5 mins | ⏳ TODO |
| Verify outputs | 10 mins | ⏳ TODO |
| **TOTAL** | **90 mins** | **Ready to start** |

---

## ✅ SUCCESS CRITERIA

After following this guide, you should have:

1. ✅ `XorZen_BETA-v-0.2.5_win64_backend.dll` built successfully
2. ✅ `xorzen_train.exe` runs without crashing
3. ✅ `xorzen_infer.exe` runs without crashing
4. ✅ No compilation errors
5. ✅ No linker errors

**If all 5 are ✅ → YOU'RE READY FOR THE NEXT PHASE (testing & optimization)**

---

## 🚀 NEXT PHASE PREVIEW

Once build works:

**Week 1**: Correctness testing
- Add unit tests
- Compare vs Python outputs
- Fix numerical bugs

**Week 2**: Performance profiling
- Identify bottlenecks
- Measure baseline speed
- Plan optimization strategy

**Week 3-4**: Optimization
- Add SIMD kernels
- Multi-threading
- CUDA (if needed)

**Month 2**: Production
- Studio integration
- Deployment testing
- Documentation

---

**LET'S GO! Start with Option 1 (MSVC) for the most reliable path.**

---

*This guide assumes no prior experience with Visual Studio. Follow step-by-step and you'll have a working build in ~90 minutes.*
