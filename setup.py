from hashlib import sha256
import os
import pathlib
from setuptools import setup
from torch.utils.cpp_extension import load

import torch.version
import sys

BASE_PATH = pathlib.Path(__file__).parent
KERNELS_REPO = BASE_PATH / "ltx_video" / "models" / "autoencoders" / "vae_patcher" / "kernels"

cu_str = ('cpu' if torch.version.cuda is None else
                  f'cu{torch.version.cuda.replace(".", "")}')
python_version = f'py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, "abiflags", "")}'

dependencies = [
    "torch>=2.1.0",
    "diffusers>=0.28.2",
    "transformers>=4.47.2,<4.52.0",
    "sentencepiece>=0.1.96",
    "huggingface-hub~=0.30",
    "einops",
    "timm"
] if not os.getenv("RUNWARE_LTX_BUILD_STAGE") else []
print(f"{dependencies=}")

home_dir = pathlib.Path(os.path.expanduser("~"))
build_repo_dir = pathlib.Path(
    os.getenv("RUNWARE_LTX_TORCH_COMPILE_DIR")  # mutable path for prod build
    or os.getenv("TORCH_EXTENSIONS_DIR")  # default path specified globally
    or home_dir / ".cache" / "torch_extensions"  # default extension path, to avoid recompilation for dev
) / f"{python_version}_{cu_str}"



def compile_object(name: str, src: list[str]):
    build_directory=build_repo_dir / name
    stored_hash = None
    pathlib.Path(build_directory).mkdir(exist_ok=True, parents=True)
    stored_hash = pathlib.Path(build_directory, ".sha256")
    hasher = sha256()
    [hasher.update(pathlib.Path(i).read_bytes()) for i in src]
    stored_digest = None
    if not stored_hash or not stored_hash.exists() or (stored_digest := stored_hash.read_text()) != hasher.hexdigest():
        print(name, hasher.hexdigest(), stored_digest)
        res = load(
            name,
            sources=src,
            build_directory=build_directory,
        )
        if build_directory and stored_hash:
            stored_hash.write_text(hasher.hexdigest())
        return res


if __name__ == "__main__":
    if not os.getenv("RUNWARE_LTX_NO_COMPILE_KERNELS"):
        compile_object(
            name="pixel_norm_inplace",
            src=[
                str(KERNELS_REPO / "pixel_norm.cpp"),
                str(KERNELS_REPO / "pixel_norm_cuda.cu"),
            ],
        )
        compile_object(
            name="inplace_add",
            src=[
                str(KERNELS_REPO / "add_inplace.cpp"),
                str(KERNELS_REPO / "add_inplace_cuda.cu"),
            ],
        )
    setup(
        install_requires=dependencies
    )

