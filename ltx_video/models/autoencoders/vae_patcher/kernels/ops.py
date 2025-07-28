from hashlib import sha256
import os
import pathlib

from torch.utils.cpp_extension import load
import torch.version
import sys

cu_str = ('cpu' if torch.version.cuda is None else
                  f'cu{torch.version.cuda.replace(".", "")}')
python_version = f'py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, "abiflags", "")}'

pixel_norm = None
inplace_add = None


source_list_pixel_norm = [
    os.path.join(os.path.dirname(__file__), "pixel_norm.cpp"),
    os.path.join(os.path.dirname(__file__), "pixel_norm_cuda.cu"),
]

source_list_inplace_add = [
    os.path.join(os.path.dirname(__file__), "add_inplace.cpp"),
    os.path.join(os.path.dirname(__file__), "add_inplace_cuda.cu"),
]

home_dir = pathlib.Path(os.path.expanduser("~"))
build_repo_dir = pathlib.Path(
    os.getenv("RUNWARE_LTX_TORCH_COMPILE_DIR")  # mutable path for prod build
    or os.getenv("TORCH_EXTENSIONS_DIR")  # default path specified globally
    or home_dir / ".cache" / "torch_extensions"  # default extension path, to avoid recompilation for dev
) / f"{python_version}_{cu_str}"

build_repo_dir.mkdir(parents=True, exist_ok=True)  # avoid nvcc crash

if not os.getenv("RUNWARE_LTX_NO_COMPILE_KERNELS"):
    for (name, src_l) in [
        ("pixel_norm", source_list_pixel_norm),
        ("inplace_add", source_list_inplace_add)
    ]:
        hasher = sha256()
        for f in src_l:
            hasher.update(pathlib.Path(f).read_bytes())
        hd = hasher.hexdigest()
        if not (saved_sha := build_repo_dir / name / ".sha256").exists() or saved_sha.read_text() != hd:
            so = load(
                name=name,
                sources=src_l,
                build_directory=build_repo_dir,
            )
            saved_sha.write_text(hd)
        else:
            import torch.ops
            so = torch.ops.load_library(build_repo_dir / name / f"{name}.so")
        setattr(sys.modules[__name__], name, so)  # save globally


def pixel_norm_inplace(x, scale, shift, eps=1e-5):
    return pixel_norm.pixel_norm_inplace(x, scale, shift, eps)  # type: ignore - guaranteed to be there


def add_inplace(x, workspace, offset):
    return inplace_add.inplace_add(x, workspace, offset)  # type: ignore - guaranteed to be there
