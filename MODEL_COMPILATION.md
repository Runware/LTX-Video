# LTX Video CUDA Kernel Compilation

This module provides automatic compilation and loading of custom CUDA kernels for LTX Video operations, including pixel normalization and inplace addition operations.

## Overview

The kernel compilation system automatically compiles CUDA kernels on first use and caches them for subsequent runs. It uses checksums to detect changes and recompile when necessary.

## Environment Variables

### `RUNWARE_LTX_TORCH_COMPILE_DIR`
Sets the primary directory for storing compiled kernels.
```bash
export RUNWARE_LTX_TORCH_COMPILE_DIR="/custom/cache/path"
```

### `TORCH_EXTENSIONS_DIR`
Fallback directory if `RUNWARE_LTX_TORCH_COMPILE_DIR` is not set.
```bash
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"
```

## Default Cache Location

If no environment variables are set, kernels are cached at:
```
~/.cache/torch_extensions/{python_version}_{cuda_version}/
```

Where:
- `python_version`: Format `py{major}{minor}{abiflags}` (e.g., `py311`)
- `cuda_version`: Format `cu{version}` (e.g., `cu118`) or `cpu`

### `RUNWARE_LTX_NO_COMPILE_KERNELS`
Disables automatic kernel compilation when set to any value.
```bash
export RUNWARE_LTX_NO_COMPILE_KERNELS=1  # Disable compilation
```

## How Kernel Loading Works

1. **Automatic Compilation**: Kernels are compiled automatically when the module is imported (unless `RUNWARE_LTX_NO_COMPILE_KERNELS` is set)

2. **Checksum Validation**: The system creates SHA256 checksums of source files and compiled binaries to detect changes

3. **Caching**: Compiled kernels are stored with their checksums to avoid unnecessary recompilation

4. **Dynamic Loading**: Kernels are attached to the module namespace dynamically after compilation

### Manual Compilation
```python
from ltx_video.models.autoencoders.vae_patcher.kernels.compiler import compiled_kernel
import pathlib

# Compile a specific kernel
kernel = compiled_kernel(
    name="pixel_norm",
    sources=["pixel_norm.cpp", "pixel_norm_cuda.cu"],
    output_directory=pathlib.Path("/custom/output/dir")
)
```

### Disabling Automatic Compilation
```python
import os
os.environ["RUNWARE_LTX_NO_COMPILE_KERNELS"] = "1"

# Now import won't trigger compilation
from ltx_video.models.autoencoders.vae_patcher.kernels.ops import compile_and_attach_kernels

# Manually compile when needed
compile_and_attach_kernels()
```

## Cache Structure

The compilation system creates the following structure:
```
{cache_dir}/
├── pixel_norm/
│   ├── pixel_norm.so
│   └── pixel_norm.sha256
└── inplace_add/
    ├── inplace_add.so
    └── inplace_add.sha256
```

## Requirements

- PyTorch with CUDA support
- CUDA toolkit installed
- C++ compiler (gcc/clang)
- NVCC (NVIDIA CUDA Compiler)

## Development

To add new kernels, update the `SOURCES` dictionary in `compiler.py`:
```python
SOURCES = {
    "new_kernel": [
        "path/to/new_kernel.cpp",
        "path/to/new_kernel_cuda.cu"
    ]
}
```
```
