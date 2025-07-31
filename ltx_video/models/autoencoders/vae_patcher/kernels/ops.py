import os
import sys

from .compiler import SOURCES, CompiledLibrary

pixel_norm = None
inplace_add = None


def compile_and_attach_kernels():
    for name, source_files in SOURCES.items():
        kernel = CompiledLibrary(name=name, sources=source_files, can_rebuild=False)
        setattr(sys.modules[__name__], name, kernel.library)


if not os.getenv("RUNWARE_LTX_NO_COMPILE_KERNELS"):
    compile_and_attach_kernels()


def pixel_norm_inplace(x, scale, shift, eps=1e-5):
    return pixel_norm.pixel_norm_inplace(x, scale, shift, eps)  # type: ignore - guaranteed to be there


def add_inplace(x, workspace, offset):
    return inplace_add.inplace_add(x, workspace, offset)  # type: ignore - guaranteed to be there
