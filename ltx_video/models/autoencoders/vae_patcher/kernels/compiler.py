import pathlib
import os
import sys
import time
import logging
import traceback
from hashlib import sha256

from torch.utils.cpp_extension import (
    load,
    get_compiler_abi_compatibility_and_version,
    get_cxx_compiler,
    get_default_build_root,
)
import torch.ops
import torch.version
from functools import cached_property

version_info = sys.version_info
cuda_version = (
    "cpu" if torch.version.cuda is None else f'cu{torch.version.cuda.replace(".", "")}'
)
python_version = f"py{version_info.major}{version_info.minor}{sys.abiflags}"

home_dir = pathlib.Path(os.path.expanduser("~"))

KERNELS_PATH = (
    pathlib.Path(
        os.getenv("RUNWARE_LTX_TORCH_COMPILE_DIR")
        or os.getenv("TORCH_EXTENSIONS_DIR")
        or get_default_build_root()
    )
    / f"{python_version}_{cuda_version}"
)

SOURCES = {
    "pixel_norm": [
        os.path.join(os.path.dirname(__file__), "pixel_norm.cpp"),
        os.path.join(os.path.dirname(__file__), "pixel_norm_cuda.cu"),
    ],
    "inplace_add": [
        os.path.join(os.path.dirname(__file__), "add_inplace.cpp"),
        os.path.join(os.path.dirname(__file__), "add_inplace_cuda.cu"),
    ],
}

logger = logging.getLogger("null")


class CompiledLibrary:
    name: str
    sources: list[str]
    kernel_directory: pathlib.Path
    output_file: pathlib.Path
    checksum_file: pathlib.Path

    def __init__(
        self,
        name: str,
        sources: list[str],
        output_directory: pathlib.Path = KERNELS_PATH,
        can_rebuild: bool = True,
        force_rebuild: bool = False,
    ):
        if not can_rebuild and force_rebuild:
            raise ValueError(
                "Cannot rebuild kernel when can_rebuild is set to False and force_rebuild is True."
            )

        if len(sources) == 0:
            raise ValueError("sources must not be empty.")

        self.name = name
        self.sources = sources
        self.kernel_directory = pathlib.Path(output_directory / name)
        self.output_file = self.kernel_directory / f"{name}.so"
        self.checksum_file = self.kernel_directory.with_suffix(".sha256")
        self.can_build = can_rebuild
        self.force_build = force_rebuild

    @cached_property
    def library(self):
        if self.already_compiled():
            logger.info(
                f"Kernel {self.name} is already compiled and up-to-date at {self.output_file}."
            )
            return self.compiled_library()
        else:
            logger.debug(f"Compiling kernel {self.name} with sources: {self.sources}")
            return self.compile()

    @property
    def checksum(self) -> str | None:
        if not self.checksum_file.exists():
            return None
        return self.checksum_file.read_text()

    @cached_property
    def expected_sources_checksum(self) -> str:
        return self.expected_checksum_hasher.hexdigest()

    @property
    def expected_checksum(self) -> str | None:
        expected_checksum_hasher = self.expected_checksum_hasher

        if self.output_file.exists():
            expected_checksum_hasher.update(self.output_file.read_bytes())
            return expected_checksum_hasher.hexdigest()
        else:
            return None

    @property
    def expected_checksum_hasher(self):
        return self._expected_checksum_hasher.copy()

    @cached_property
    def _expected_checksum_hasher(self):
        expected_checksum_hasher = sha256()

        for source in self.sources:
            bytes = pathlib.Path(source).read_bytes()
            expected_checksum_hasher.update(bytes)

        return expected_checksum_hasher

    def already_compiled(self) -> bool:
        if self.force_build:
            logger.debug(
                f"Force rebuild is enabled. Will compile kernel {self.name} from sources."
            )
            return False

        expected = self.expected_checksum
        existing_checksum = self.checksum

        if expected is None:
            logger.debug(
                f"Output file {self.output_file} does not exist. Will compile from sources. Checksums will not match."
            )
            return False

        if existing_checksum is None:
            logger.debug(
                f"Checksum file {self.checksum_file} does not exist. Will compile from sources. Checksums will not match."
            )
            return False

        if existing_checksum == expected:
            return True

        logger.warning(
            f"Kernel {self.name} is not up-to-date. Expected checksum: {expected}, existing checksum: {existing_checksum}. Will compile from sources most likely."
        )
        return False

    def compile(self):
        self.ensure_can_compile()

        self.kernel_directory.mkdir(exist_ok=True, parents=True)

        if self.force_build and self.output_file.exists():
            logger.debug(
                f"Removing existing output file {self.output_file} due to force rebuild."
            )
            self.output_file.unlink()

        start_time = time.time()
        res = load(
            self.name,
            sources=self.sources,
            build_directory=f"{self.kernel_directory.absolute()}",
        )
        logger.debug(
            f"Kernel {self.name} compiled in {time.time() - start_time:.2f} seconds"
        )

        if not self.output_file.exists():
            raise RuntimeError(
                f"Failed to compile kernel {self.name}. Output file {self.output_file} does not exist."
            )

        hasher = self.expected_checksum_hasher
        hasher.update(self.output_file.read_bytes())
        self.checksum_file.write_text(hasher.hexdigest())

        return res

    def ensure_can_compile(self):
        if not self.can_build:
            raise RuntimeError(
                f"Kernel {self.name} is not compiled and allow_rebuild is set to False. Cannot compile."
            )

        status, version = get_compiler_abi_compatibility_and_version(get_cxx_compiler())
        if not status:
            raise RuntimeError(
                f"Compiler is not compatible with the current platform. Cannot compile kernel {self.name}."
            )

        logger.info(f"Compiler ABI compatibility status: {status}, version: {version}")

    def compiled_library(self):
        if not self.output_file.exists():
            raise RuntimeError(
                f"Kernel {self.name} is not compiled. Output file {self.output_file} does not exist."
            )

        torch.ops.load_library(self.output_file)
        return torch.ops.__getattr__(self.name)


def compile_all_kernels(
    kernels_directory: pathlib.Path = KERNELS_PATH, force_rebuild: bool = False
):
    if not kernels_directory.exists():
        kernels_directory.mkdir(exist_ok=True, parents=True)

    for name, sources in SOURCES.items():
        try:
            kernel = CompiledLibrary(
                name=name,
                sources=sources,
                output_directory=kernels_directory,
                can_rebuild=True,
                force_rebuild=force_rebuild,
            )
            setattr(sys.modules[__name__], name, kernel.library)
        except Exception as e:
            logger.error(f"Failed to compile kernel {name}: {e}")
            raise


def load_all_kernels(kernels_directory: pathlib.Path = KERNELS_PATH):
    if not kernels_directory.exists():
        raise RuntimeError(
            f"Kernels directory {kernels_directory} does not exist. Cannot load kernels."
        )

    failed_kernels = {}

    for name, sources in SOURCES.items():
        try:
            CompiledLibrary(
                name=name,
                sources=sources,
                output_directory=kernels_directory,
                can_rebuild=False,
            ).library
        except Exception as e:
            trace = traceback.format_exc()
            failed_kernels[name] = {
                "error": str(e),
                "type": type(e).__name__,
                "traceback": trace,
            }

    if len(failed_kernels) > 0:
        error_details = []
        for kernel_name, error_info in failed_kernels.items():
            error_details.append(f"Kernel '{kernel_name}':\n{error_info['traceback']}")

        raise RuntimeError("Failed to load kernels:\n" + "\n".join(error_details))
