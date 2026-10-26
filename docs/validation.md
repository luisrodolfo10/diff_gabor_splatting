# Validation of the initial implementation

Validated locally on 2026-10-26 with Python 3.12, PyTorch 2.8.0+cu128, the system
CUDA 12.0 compiler, GCC 13, and an NVIDIA GeForce RTX 3060 Laptop GPU (SM 8.6).
PyTorch reports a CUDA minor-version mismatch during compilation; the source
build and runtime checks nevertheless pass in this environment. This is the
validated combination, not a claim that every Torch/CUDA combination works.
The system Python headers were absent, so the build was given the existing
Python 3.12 development include directory through CPLUS_INCLUDE_PATH.

The full local suite passed 42 tests, including both optional upstream comparisons.

Checks performed:

- Ruff lint and format checks and Python syntax compilation.
- Independent float64 PyTorch surface-intersection, appearance, and compositor
  outputs and derivatives for both methods and all differentiable maps.
- Central finite differences for representative geometry and appearance parameters.
- Fused/loop and packed/dense image and gradient equivalence, multiple scenes and
  cameras, arbitrary leading scene dimensions, and a non-default CUDA stream.
- Empty/invisible scenes, SH degrees 0–3, zero/saturated opacity, early termination,
  subpixel surfaces, degenerate conics, explicit input errors, and global-default
  dtype independence.
- Optional compiled upstream output comparisons at the commits in NOTICE.md,
  including radii and all supplied surface maps across multiple image tiles.
- Both synthetic optimization examples reduce their losses over 30 steps.
- Editable installation, source archive and wheel builds, and wheel installation
  into a separate environment with an import path outside this checkout.
- Checkout and consuming-project installation with uv 0.12.5, isolated build
  dependencies, and matching Torch 2.8.0+cu128; see [installation.md](installation.md).

To reproduce the optional upstream checks, separately build the two pinned
rasterizers, then add their checkout directories to PYTHONPATH when running
`pytest tests/test_upstream.py`. These packages are validation-only dependencies;
they are not required by gabor-splatting. The ordinary suite skips these two
comparisons when upstream packages are absent. No alternate renderer is used at
runtime.

The default compact benchmark was run with five warmups and 20 measured iterations
for 10,000 moderately tilted primitives, one scene, four cameras, and 256×256 images.
Its raw environment, workload, timings, and memory results are in
[rtx3060_torch28.json](../benchmarks/results/rtx3060_torch28.json).

The fused dense forward/backward workload took about 17.8 ms for classic Gabor
versus 38.1 ms for the single-view loop (2.14x), and 21.7 ms for neural Gabor versus
43.3 ms for the loop (1.99x). These are local synthetic measurements, including
validation and autograd overhead; they do not establish reconstruction quality or
a universal speedup. Packing is workload-dependent and still projects dense pairs
before compaction. The checkout had no Git repository, so the recorded revision
is null.

The locally generated wheel targets CPython 3.12, Linux x86-64, and SM 8.6; it is
not a universal wheel. Rebuild from source for a different Python, Torch/CUDA
stack, or GPU architecture. No package registry publication was performed.
