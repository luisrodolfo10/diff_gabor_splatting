# Installation

The distribution is `gabor-splatting`; the import is `gabor_splatting`. Classic
and neural Gabor rendering are installed together in one CUDA extension.

## Requirements

Use Linux, Python 3.10+, matching Python development headers, an NVIDIA GPU,
CUDA-enabled PyTorch, a CUDA toolkit containing `nvcc`, and a compatible C++
compiler. [Validation](validation.md) records the tested stack. A PyTorch CUDA
runtime alone does not provide the compiler toolkit needed to build an extension.

## Install from Git

In a consuming project's `pyproject.toml`, pin the release and provide matching
build/runtime Torch requirements:

```toml
[project]
name = "my-gabor-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "torch==2.8.*",
  "gabor-splatting @ git+https://github.com/luisrodolfo10/diff_gabor_splatting.git@v0.1.1",
]

[tool.uv.extra-build-dependencies]
gabor-splatting = [{ requirement = "torch", match-runtime = true }]

[tool.uv.sources]
torch = { index = "pytorch-cu128" }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

Then run:

```bash
uv sync
uv run python -c "from gabor_splatting import rasterize_gabors, rasterize_neural_gabors"
```

Choose the Torch constraint and CUDA index for your application. The package
declares setuptools, wheel, Ninja, and Torch as build requirements; `match-runtime`
keeps the build's Torch aligned with the application. This follows uv's
[per-package build configuration](https://docs.astral.sh/uv/concepts/projects/config/#augmenting-build-dependencies).

The public HTTPS URL needs no GitHub SSH key. An authenticated SSH source can
instead use `git+ssh://git@github.com/luisrodolfo10/diff_gabor_splatting.git@v0.1.1`.

## Work from a checkout

```bash
git clone https://github.com/luisrodolfo10/diff_gabor_splatting.git
cd diff_gabor_splatting
uv sync
```

The checkout selects Torch 2.8 with the cu128 index and matches Torch at build
time. It targets Python 3.12 for development; the library supports Python 3.10+.
For examples and validation tools:

```bash
uv sync --all-extras
uv run python examples/fit_texture.py --method both --steps 100
uv run pytest -q
```

`dev` supplies validation tools and `examples` supplies Pillow for PNG export.
Use `uv sync --extra examples` when only image export is needed.

## Build settings

When toolkit discovery or target architectures need explicit settings:

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=4 uv sync
```

Use the architecture for your GPU. The extension compiles at installation;
there is no compilation on import. Rebuild after changing Torch or CUDA:

```bash
uv sync --reinstall-package gabor-splatting
```

## Troubleshooting

- **`Python.h` missing:** install development headers matching the selected Python.
  A nonstandard header directory can be supplied through `CPLUS_INCLUDE_PATH`.
- **`nvcc` missing:** install a CUDA development toolkit and set `CUDA_HOME` if
  needed. A working `nvidia-smi` does not establish that the compiler is installed.
- **Compiler/version errors:** select a host compiler supported by the toolkit
  and a compatible CUDA-enabled Torch build.
- **Import errors after upgrading Torch:** rebuild with the command above.

Source and wheel distributions can be created with `uv build`. Wheels are specific
to their Python, platform, Torch/CUDA stack, and compiled GPU architectures.
