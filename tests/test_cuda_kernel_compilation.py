import pytest
import pathlib
import os
import tempfile
import shutil
from hashlib import sha256
from pathlib import Path

from unittest.mock import Mock, patch
from torch.utils.cpp_extension import (
    get_compiler_abi_compatibility_and_version,
    _check_cuda_version,
    get_cxx_compiler,
)
from ltx_video.models.autoencoders.vae_patcher.kernels.compiler import CompiledLibrary
import subprocess

if os.environ.get("TORCH_CUDA_ARCH_LIST") is None:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0"


@pytest.fixture(scope="session")
def compilation_fixtures() -> Path:
    return Path(__file__).parent / "fixtures" / "compilation"


@pytest.fixture
def test_kernel_sources(compilation_fixtures):
    sources = [
        compilation_fixtures / "kernel.cpp",
        compilation_fixtures / "kernel_cuda.cu",
    ]
    return sources


class TestCompiledLibrary:

    @pytest.fixture
    def temp_dir(self):
        temp_dir = pathlib.Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_compiled_library(self):
        mock_lib = Mock()
        mock_lib.test_function = Mock(return_value="test_result")
        return mock_lib

    def test_init_basic(self, temp_dir, test_kernel_sources):
        lib = CompiledLibrary(
            name="test_kernel", sources=test_kernel_sources, output_directory=temp_dir
        )

        assert lib.name == "test_kernel"
        assert lib.sources == test_kernel_sources
        assert lib.kernel_directory == temp_dir / "test_kernel"
        assert lib.output_file == temp_dir / "test_kernel" / "test_kernel.so"
        assert lib.checksum_file == (temp_dir / "test_kernel").with_suffix(".sha256")
        assert lib.can_build is True
        assert lib.force_build is False

    def test_init_with_custom_arguments(self, test_kernel_sources):
        lib = CompiledLibrary(
            name="custom_kernel",
            sources=test_kernel_sources,
            can_rebuild=False,
            force_rebuild=False,
        )

        assert lib.can_build is False
        assert lib.force_build is False

    def test_init_conflicting_arguments_args(self, test_kernel_sources):
        with pytest.raises(
            ValueError,
            match="Cannot rebuild kernel when can_rebuild is set to False and force_rebuild is True.",
        ):
            CompiledLibrary(
                name="test_kernel",
                sources=test_kernel_sources,
                can_rebuild=False,
                force_rebuild=True,
            )

    def test_expected_sources_checksum(self, temp_dir, test_kernel_sources):
        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)

        expected_hasher = sha256()
        for source in test_kernel_sources:
            expected_hasher.update(pathlib.Path(source).read_bytes())
        expected = expected_hasher.hexdigest()

        assert lib.expected_sources_checksum == expected

    def test_expected_checksum_hasher_copy(self, temp_dir, test_kernel_sources):
        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)

        hasher1 = lib.expected_checksum_hasher
        hasher2 = lib.expected_checksum_hasher

        assert hasher1 is not hasher2
        assert hasher1.hexdigest() == hasher2.hexdigest()

    def test_checksum_file_not_exists(self, temp_dir, test_kernel_sources):
        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)
        assert lib.checksum is None

    def test_checksum_file_exists(self, temp_dir, test_kernel_sources):
        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)

        test_checksum = "abc123"
        lib.checksum_file.parent.mkdir(parents=True, exist_ok=True)
        lib.checksum_file.write_text(test_checksum)

        assert lib.checksum == test_checksum

    def test_expected_checksum_no_output_file(self, temp_dir, test_kernel_sources):
        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)
        assert lib.expected_checksum is None

    def test_expected_checksum_with_output_file(self, temp_dir, test_kernel_sources):
        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)

        lib.output_file.parent.mkdir(parents=True, exist_ok=True)
        lib.output_file.write_bytes(b"mock compiled library")

        expected_hasher = sha256()
        for source in test_kernel_sources:
            expected_hasher.update(pathlib.Path(source).read_bytes())
        expected_hasher.update(lib.output_file.read_bytes())

        assert lib.expected_checksum == expected_hasher.hexdigest()

    def test_already_compiled_no_output_file(self, temp_dir, test_kernel_sources):
        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)
        assert lib.already_compiled() is False

    def test_already_compiled_no_checksum_file(self, temp_dir, test_kernel_sources):
        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)

        lib.output_file.parent.mkdir(parents=True, exist_ok=True)
        lib.output_file.write_bytes(b"mock library")

        assert lib.already_compiled() is False

    def test_already_compiled_checksum_mismatch(self, temp_dir, test_kernel_sources):
        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)

        lib.output_file.parent.mkdir(parents=True, exist_ok=True)
        lib.output_file.write_bytes(b"mock library")
        lib.checksum_file.write_text("wrong_checksum")

        assert lib.already_compiled() is False

    def test_already_compiled_checksum_match(self, temp_dir, test_kernel_sources):
        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)

        lib.output_file.parent.mkdir(parents=True, exist_ok=True)
        lib.output_file.write_bytes(b"mock library")

        expected_hasher = sha256()
        for source in test_kernel_sources:
            expected_hasher.update(pathlib.Path(source).read_bytes())
        expected_hasher.update(lib.output_file.read_bytes())
        lib.checksum_file.write_text(expected_hasher.hexdigest())

        assert lib.already_compiled() is True

    def test_ensure_can_compile_not_allowed(self, temp_dir, test_kernel_sources):
        lib = CompiledLibrary("test", test_kernel_sources, temp_dir, can_rebuild=False)

        with pytest.raises(
            RuntimeError, match="is not compiled and allow_rebuild is set to False"
        ):
            lib.ensure_can_compile()

    def test_ensure_can_compile_success(self, temp_dir, test_kernel_sources):
        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)

        lib.ensure_can_compile()

    @patch("ltx_video.models.autoencoders.vae_patcher.kernels.compiler.load")
    def test_compile_success(self, mock_load, temp_dir, test_kernel_sources):

        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)

        mock_load.side_effect = lambda *args, **kwargs: (
            lib.output_file.parent.mkdir(parents=True, exist_ok=True),
            lib.output_file.write_bytes(b"mock compiled library"),
            Mock(),
        )[-1]

        lib.compile()

        assert lib.output_file.exists()
        assert lib.checksum_file.exists()

    @patch("ltx_video.models.autoencoders.vae_patcher.kernels.compiler.load")
    @patch("pathlib.Path.exists")
    def test_compile_output_file_not_created(
        self, mock_load, mock_exists, temp_dir, test_kernel_sources
    ):

        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)
        mock_exists.return_value = False
        mock_load.return_value = None

        with pytest.raises(RuntimeError, match="Failed to compile kernel test"):
            lib.compile()
        mock_load.assert_called_once()

    @patch("torch.ops.load_library")
    @patch("torch.ops.__getattr__")
    def test_compiled_library_success(
        self,
        mock_getattr,
        mock_load_library,
        temp_dir,
        test_kernel_sources,
        mock_compiled_library,
    ):
        mock_getattr.return_value = mock_compiled_library

        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)

        lib.output_file.parent.mkdir(parents=True, exist_ok=True)
        lib.output_file.write_bytes(b"mock library")

        result = lib.compiled_library()

        assert result == mock_compiled_library
        mock_load_library.assert_called_once_with(lib.output_file)
        mock_getattr.assert_called_once_with("test")

    def test_compiled_library_no_output_file(self, temp_dir, test_kernel_sources):
        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)

        with pytest.raises(RuntimeError, match="Kernel test is not compiled"):
            lib.compiled_library()

    @patch("torch.utils.cpp_extension.load")
    @patch("torch.ops.load_library")
    @patch("torch.ops.__getattr__")
    def test_library_property_already_compiled(
        self,
        mock_getattr,
        mock_load_library,
        mock_load,
        temp_dir,
        test_kernel_sources,
        mock_compiled_library,
    ):
        mock_getattr.return_value = mock_compiled_library

        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)

        lib.output_file.parent.mkdir(parents=True, exist_ok=True)
        lib.output_file.write_bytes(b"mock library")

        expected_hasher = sha256()
        for source in test_kernel_sources:
            expected_hasher.update(pathlib.Path(source).read_bytes())
        expected_hasher.update(lib.output_file.read_bytes())
        lib.checksum_file.write_text(expected_hasher.hexdigest())

        result = lib.library

        assert result == mock_compiled_library
        mock_load.assert_not_called()
        mock_load_library.assert_called_once()

    @patch("torch.utils.cpp_extension.load")
    def test_library_property_needs_compilation(
        self, mock_load, temp_dir, test_kernel_sources, mock_compiled_library
    ):
        mock_load.return_value = mock_compiled_library

        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)

        with patch.object(
            lib, "compile", return_value=mock_compiled_library
        ) as mock_compile:
            result = lib.library

            assert result == mock_compiled_library
            mock_compile.assert_called_once()

    @patch("torch.utils.cpp_extension.load")
    def test_library_property_cached(
        self, mock_load, temp_dir, test_kernel_sources, mock_compiled_library
    ):
        mock_load.return_value = mock_compiled_library

        lib = CompiledLibrary("test", test_kernel_sources, temp_dir)

        with patch.object(
            lib, "compile", return_value=mock_compiled_library
        ) as mock_compile:
            result1 = lib.library
            result2 = lib.library

            assert result1 is result2
            mock_compile.assert_called_once()


class TestCompiledLibraryIntegration:

    @pytest.fixture
    def integration_temp_dir(self):
        temp_dir = pathlib.Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_full_workflow_missing_sources(self, integration_temp_dir):
        non_existent_sources = [
            str(integration_temp_dir / "missing.cpp"),
            str(integration_temp_dir / "missing.cu"),
        ]

        lib = CompiledLibrary("test", non_existent_sources, integration_temp_dir)

        with pytest.raises(FileNotFoundError):
            _ = lib.expected_sources_checksum

    @pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
    def test_real_compilation_environment(
        self, integration_temp_dir, test_kernel_sources
    ):
        lib = CompiledLibrary(
            name="test_kernel",
            sources=test_kernel_sources,
            output_directory=integration_temp_dir,
        )

        compiled_lib = lib.library
        assert compiled_lib is not None


class TestCompiledLibraryEdgeCases:

    def test_empty_sources_list(self, tmp_path):
        with pytest.raises(Exception):
            lib = CompiledLibrary("test", [], tmp_path)
            _ = lib.expected_sources_checksum

    def test_permission_denied_output_directory(self):
        read_only_dir = pathlib.Path("/root")  # Typically not writable
        if read_only_dir.exists() and not os.access(read_only_dir, os.W_OK):
            sources = ["test.cpp"]
            lib = CompiledLibrary("test", sources, read_only_dir)

            with patch("torch.cuda.is_available", return_value=True):
                with pytest.raises(Exception):
                    lib.compile()

    def test_concurrent_access(self, tmp_path):
        source_file = tmp_path / "test.cpp"
        source_file.write_text("// test")

        lib = CompiledLibrary("test", [str(source_file)], tmp_path)

        checksum1 = lib.expected_sources_checksum
        checksum2 = lib.expected_sources_checksum

        assert checksum1 == checksum2
        assert checksum1 is checksum2


def cxx_compiler_major_version():
    try:
        output = subprocess.check_output(
            [get_cxx_compiler(), "--version"], stderr=subprocess.STDOUT
        )
        version_line = output.decode().splitlines()[0]
        version = int(version_line.split()[2].split(".")[0])
        return version
    except Exception as e:
        print(f"Error getting GCC version: {e}")
        return None


def cuda_available():
    cxx_major_version = cxx_compiler_major_version()
    if not cxx_major_version:
        return False
    if cxx_major_version > 14:
        return False
    try:
        ok, torch_version = get_compiler_abi_compatibility_and_version(
            get_cxx_compiler()
        )
        if not ok:
            return False

        _check_cuda_version(get_cxx_compiler(), torch_version)
        return True

    except:
        return False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--disable-warnings"])
