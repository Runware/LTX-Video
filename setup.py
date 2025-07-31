from setuptools import setup

dependencies = [
    "torch>=2.1.0",
    "diffusers>=0.28.2",
    "transformers>=4.47.2,<4.52.0",
    "sentencepiece>=0.1.96",
    "huggingface-hub~=0.30",
    "einops",
    "timm",
    "ninja", # For compiling custom C++/CUDA extensions
]


if __name__ == "__main__":
    setup(
        install_requires=dependencies
    )
