import argparse
import os
from pathlib import Path
import ltx_video.models.autoencoders.vae_patcher.kernels.compiler as compiler


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="LTX Video utility commands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Enable verbose output"
    )

    commands = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    add_compile_kernels_command(commands)
    add_load_all_kernels_command(commands)

    return parser.parse_args()


def add_compile_kernels_command(commands):

    compile_parser = commands.add_parser("compile-kernels", help="Compile CUDA kernels")

    compile_parser.add_argument(
        "--arch-list",
        type=str,
        default=os.environ.get("TORCH_CUDA_ARCH_LIST", "8.6;9.0"),
        help="CUDA architecture list (semicolon-separated)",
    )

    compile_parser.add_argument(
        "--kernels-dir",
        type=Path,
        default=Path(compiler.KERNELS_PATH),
        help="Output directory for compiled kernels",
    )

    compile_parser.add_argument(
        "--force-rebuild",
        action="store_true",
        default=False,
        help="Force rebuild of all kernels, even if they are already compiled",
    )


def add_load_all_kernels_command(commands):
    load_parser = commands.add_parser(
        "load-all-kernels", help="Load all compiled CUDA kernels"
    )

    load_parser.add_argument(
        "--kernels-dir",
        type=Path,
        default=Path(compiler.KERNELS_PATH),
        help="Output directory for compiled kernels",
    )


def build_console_logger():
    import logging

    logger = logging.getLogger("ltx_video")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger


def main():
    args = parse_command_line()

    if args.verbose:
        compiler.logger = build_console_logger()

    if args.command == "compile-kernels":
        os.environ["TORCH_CUDA_ARCH_LIST"] = args.arch_list
        compiler.compile_all_kernels(
            kernels_directory=args.kernels_dir, force_rebuild=args.force_rebuild
        )
    elif args.command == "load-all-kernels":
        compiler.load_all_kernels(kernels_directory=args.kernels_dir)


if __name__ == "__main__":
    main()
