#!/usr/bin/env python3
"""Fix all __restrict__ to RESTRICT in C++ files"""

files = [
    r"C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen\speed\csrc\attention_ops.cpp",
    r"C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen\speed\csrc\expert_dispatch.cpp",
    r"C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen\speed\csrc\math_utils.cpp",
    r"C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen\speed\csrc\router_ops.cpp",
    r"C:\Users\akikf\programing\ai\xorzen_0.2.4\xorzen\speed\csrc\ssm_scan.cpp",
]

macro = """
// Cross-platform restrict keyword
#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#else
    #define RESTRICT
#endif
"""

for filepath in files:
    print(f"Processing: {filepath}")
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Add macro after #include "xorzen_kernels.h"
    if macro.strip() not in content:
        content = content.replace(
            '#include "xorzen_kernels.h"',
            f'#include "xorzen_kernels.h"{macro}'
        )
    
    # Replace all __restrict__
    content = content.replace('__restrict__', 'RESTRICT')
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  ✓ Fixed")

print("\nAll C++ files fixed!")
