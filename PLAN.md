# Gabor splatting package plan

Status: implemented for the initial renderer scope. See README.md, docs/contracts.md,
and docs/validation.md for the delivered API, numerical contracts, and validation.

## Scope and sources

Create one installable PyTorch/CUDA rasterization library for classic and neural
Gabor splatting, with a standard source-layout package and fused scene/camera
batching inspired by [gsplat](https://docs.gsplat.studio/main/).

Assumed methods:

- Classic: https://github.com/haato-w/3d-gabor-splatting and its
  https://github.com/haato-w/diff-gabor-rasterization rasterizer.
- Neural: https://github.com/haato-w/neural-gabor-splatting and its
  https://github.com/haato-w/diff-neural-gabor-rasterization rasterizer.

These methods use oriented 2D Gaussian surface primitives in 3D. Preserve their
local-coordinate appearance evaluation and surface projection. Share batching
infrastructure while preserving each method's mathematical contract.

The first release is a differentiable renderer. Full scene training, COLMAP IO,
mesh extraction, frequency-aware densification, and checkpoint conversion remain
outside this initial scope. Include a small synthetic optimization example.

## 1. Freeze the numerical contract

- Pin upstream commits and record attribution and applicable license files.
- Trace forward and backward contracts: coordinate frames, pixel centers,
  quaternion order, scale activation, opacity range, SH ordering, local frequency
  units, phase units, view direction, clamping, and visibility thresholds.
- Keep the classic four-frequency layout and neural 5-input, 6-hidden,
  3-output SIREN layout initially; preserve upstream activation and initialization.
- Specify which outputs and inputs have supported gradients. Preserve useful
  screen-space gradient information for downstream densification.
- Check alpha, depth, normal, and distortion outputs individually rather than
  assuming that upstream output gradients are complete or correct.
- Preserve point evaluation of appearance at surface intersections and the
  upstream low-pass filter.

Deliverable: documented formulas, shapes, conventions, and gradient support.

## 2. Package both methods together

Proposed distribution: `gabor-splatting`; import: `gabor_splatting`.
Confirm distribution-name availability before publication.

Use `pyproject.toml`, a minimal CUDA-extension `setup.py`, `MANIFEST.in`, a
`src/gabor_splatting/` layout, and one compiled `gabor_splatting._C` extension.
Keep Python validation/autograd separate from native geometry and appearance.
Use private implementation modules where useful, with direct function names and
brief function docstrings/comments.

Compile the CUDA extension at installation. Document installing
CUDA-enabled Torch and build tools through `uv sync`, using either a checkout
or the tagged Git dependency. Verify editable and source archive
installation. State the supported Torch/CUDA/compiler combinations; source
installation still needs a compatible CUDA toolkit. No import-time JIT or CPU
fallback. Prebuilt wheels are a later release task.

## 3. Keep the public API small and explicit

Export two functions:

- `rasterize_gabors(...)`
- `rasterize_neural_gabors(...)`

Both accept the same geometry/camera arguments and return
`(rgb, alpha, meta)`. Use separate explicit appearance arguments rather than a
method selector with incompatible optional inputs. Prefer names such as
`quaternions`, `view_matrices`, `intrinsics`, `frequency_weights`,
`frequency_magnitudes`, `phases`, `input_weights`, and `output_weights`.
Finalize names against the actual upstream meaning during step 1.

Shared shape contract:

- Means: `[..., N, 3]`; quaternions: `[..., N, 4]`, wxyz.
- Surface scales: `[..., N, 2]`; opacities: `[..., N]`.
- World-to-camera matrices: `[..., C, 4, 4]`.
- Pixel-space intrinsics: `[..., C, 3, 3]`.
- RGB: `[..., C, H, W, 3]`; alpha: `[..., C, H, W, 1]`.

Leading dimensions represent independent scenes; C represents cameras sharing
each scene. Initially require matching leading dimensions and a fixed N across
scenes. CUDA float32 is the initial rendering contract. Reject unsupported
devices, dtypes, layouts, and shapes explicitly. Define camera-gradient support
before advertising it.

Use named metadata entries for radii/projected means and supported surface maps;
avoid exposing an undocumented positional `allmap`. Document units, frames,
normalization, shapes, and differentiability for each entry. Avoid compatibility
classes unless a concrete consumer needs them.

## 4. Establish single-view correctness

Port each appearance evaluator and backward using a common surface rendering
pipeline where the contracts actually match. Preserve classic SH/color behavior
and neural per-primitive weight layout. Keep appearance evaluation inside the
pixel/primitive CUDA work; a once-per-primitive color prediction would change
the method.

Validate against pinned upstream renders and small independent mathematical
checks. Check geometry, opacity, classic frequencies/phases/colors, and neural
weights/biases. Use finite differences away from discrete culling, ordering, and
clamping boundaries. Do not preserve upstream defects merely for parity.

## 5. Add fused scene and camera batching

Use a gsplat-inspired image-aware batching strategy:

1. Flatten leading scene dimensions and assign each scene/camera pair an image ID.
2. Project all primitive/camera pairs together using the surface geometry.
3. Count and emit tile intersections with image-aware tile IDs.
4. Sort globally by image/tile and depth, then build tile ranges.
5. Render all images in a shared tile launch with the selected appearance kernel.
6. Accumulate backward contributions into scene primitives and appearance
   parameters shared by multiple cameras.

Start with dense pair storage. Support a single view through this same pipeline;
avoid maintaining a second legacy renderer. Add visible-pair compaction
(`packed=True`) after dense parity, sharing the compositor. Packed here means
visible camera/primitive pairs, not ragged scene inputs.

Validate fused outputs and gradients against repeated single-view calls for both
methods, multiple scenes/cameras, arbitrary leading dimensions, empty scenes,
invisible primitives, and non-default CUDA streams/devices where available.
Do not promise a speedup before measuring it. Defer selected-pixel sparse
rendering, ragged batches, arbitrary neural architectures, and channel chunking.

## 6. Build one small benchmark with explicit names

Use `benchmarks/benchmark_rasterization.py` and public package APIs only.

Suggested flags: `--method classic|neural|both`, `--num-primitives`,
`--num-scenes`, `--num-cameras`, `--image-width`, `--image-height`,
`--warmup-iterations`, `--measurement-iterations`, and `--output-json`.

Compare `single_view_loop`, `fused_dense`, and, once available, `fused_packed`.
Use a modest default workload: 10,000 primitives per scene, one scene, four
cameras, 256x256 images, five warmups, and 20 measured iterations. Users can
request larger workloads explicitly; avoid a large default sweep.

Report `forward_ms`, `backward_ms`, `forward_backward_ms`, `images_per_second`,
`peak_allocated_memory_mb`, and `speedup_vs_single_view_loop`, with exact timing
and memory definitions. Record GPU, Torch/CUDA, seed, method, workload, and
revision. Use CUDA events, independent warmup, equivalent loss reduction and
gradient reset, and exclude scene construction and IO from timings. Check
output/gradient parity before timing. The loop baseline must also render every
scene and camera; scene count and camera count must never share an ambiguous
`batch_size` label. Compare speedups within each method; different appearance
models do not imply equal image quality.

## 7. Validate and release

- Run focused numerical and batching checks, including auxiliary-output gradients
  for every output documented as differentiable.
- Build an sdist and wheel; inspect included CUDA sources, headers, and licenses.
- Install the artifact into a clean supported environment, import it, and run
  each renderer forward/backward and the benchmark smoke workload.
- Document installation, both minimal rendering examples, tensor conventions,
  gradient support, and benchmark commands. Configure cheap lint/package CI and
  GPU validation where an actual GPU runner exists.

Implementation order: numerical contract -> package and classic single-view ->
neural single-view -> fused dense batching for both -> packed batching -> compact
benchmark -> installation/release validation. Each stage must be correct before
using it as the next stage's reference.
