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

### CLI Commands

To compile kernels manually, you can use the provided CLI commands:
```bash
python -m ltx_video --help 
```

```
LTX Video utility commands

positional arguments:
  {compile-kernels,load-all-kernels}
                        Available commands
    compile-kernels     Compile CUDA kernels
    load-all-kernels    Load all compiled CUDA kernels

options:
  -h, --help            show this help message and exit
  --verbose             Enable verbose output (default: False)
```

**Kernel Compilation Command**

```bash
python -m ltx_video compile-kernels --arch-list=sm_70;sm_75 --kernels-dir=/path/to/kernels --force-rebuild
```

```
usage: python -m ltx_video compile-kernels [-h] [--arch-list ARCH_LIST] [--kernels-dir KERNELS_DIR] [--force-rebuild]

options:
  -h, --help            show this help message and exit
  --arch-list ARCH_LIST
                        CUDA architecture list (semicolon-separated)
  --kernels-dir KERNELS_DIR
                        Output directory for compiled kernels
  --force-rebuild       Force rebuild of all kernels, even if they are already compiled
```

**Testing Kernel Compilation**

After compiling the kernels, they can be loaded and tested using the following command, in order to test if the kernels are loading correctly:

```bash
usage: python -m ltx_video  load-all-kernels [-h] [--kernels-dir KERNELS_DIR]

options:
  -h, --help            show this help message and exit
  --kernels-dir KERNELS_DIR
                        Output directory for compiled kernels
```
