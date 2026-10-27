# v0.1.1 performance work

The backward compositor now cooperatively loads 64 primitive records into shared
memory per batch. One block handles one 16×16 pixel tile, starting its reverse
traversal at the block's largest accepted contributor count. A pixel contributes
zero outside its own contributor range; inactive and edge lanes still participate
in warp reductions. Each parameter receives one global atomic update per active
warp instead of one update per contributing pixel. Background gradients use the
same warp reduction. Forward uses the same shared primitive batches and accumulates
its output maps in registers before storing them.

Neural appearance gradients are reduced as they are computed. This avoids holding
all 57 appearance derivatives live at once. On the validated SM 8.6 build, the
neural backward uses 122 registers per thread, with no register spills. Classic
SH colors are evaluated together in a fused forward/backward pair of kernels;
coefficient sharing and direction/camera gradients are preserved.

`validate=False` skips numerical value checking for trusted inputs. Shape, dtype,
device, contiguity, and scalar checks remain enabled. The default is still
`validate=True`; tile-intersection allocation still requires a synchronization.
Projection remains in PyTorch.

## Profile an actual trained tile

Export the arguments at the public renderer boundary from the existing training
or profile script. Keep its image size, camera, activated primitive parameters,
SH degree, packing mode, and target image. This avoids coupling the benchmark to
an application's checkpoint layout:

```python
import torch

# inputs is the exact keyword-argument dictionary passed to the public renderer.
# target_rgb is float32 RGB in [..., C, H, W, 3] or [H, W, 3], with one camera/tile.
# trainable lists the input tensors whose gradients the training workload needs.
torch.save(
    {
        "method": "classic",  # or "neural"
        "inputs": inputs,
        "target": target_rgb,
        "trainable": trainable,
    },
    "frame100_classic_tile.pt",
)
```

The bundle must contain tensors and ordinary scalar/container values, suitable
for `torch.load(..., weights_only=True)`. Exclude `validate` from `inputs`; select
that behavior with the command-line flag. Camera/primitive dimensions follow the
[public contracts](contracts.md), and the bundle must describe exactly one image.

```bash
uv run python benchmarks/benchmark_real_scene.py frame100_classic_tile.pt --skip-validation --max-step-ms 25 --output-json classic-profile.json
uv run python benchmarks/benchmark_real_scene.py frame100_neural_tile.pt --skip-validation --max-step-ms 35 --output-json neural-profile.json
```

The measured objective is render + RGB L1 loss + backward to the specified input
tensors. It excludes loading, device copies, model activations before the renderer,
optimizer updates, and densification. Match those exclusions and the loss when
comparing it with an application's training-step profile.

`step_ms` uses CUDA events; `wall_ms_per_step` also includes host execution.
The kernel split comes from a separate CUPTI-backed PyTorch profiler trace:
`forward_kernel_ms`, `backward_kernel_ms`, `sh_kernel_ms`, and per-kernel timings.
`other_cuda_kernel_ms` includes SH, projection, binning, loss, and other kernels;
it is not a claim to measure only PyTorch work or CPU overhead. Profiled kernel
times can differ from unprofiled step times because instrumentation and GPU clocks
change execution. Missing kernel capture is an error, not a zero-time result.
Run profiles serially on an otherwise idle GPU. `--max-step-ms` saves the report
and exits unsuccessfully if the measured mean exceeds the explicit budget.

## Available validation and remaining target measurements

The frame-100 model inputs, their original profile script, the RTX 3090, and the
cu129 deployment environment were unavailable in this workspace. The requested
real-scene budgets (classic <=25 ms/step, neural <=35 ms/step at 1536×810) therefore
remain **unverified**. No claim that these budgets have been met is made here.

An explicit synthetic control was measured on the RTX 3060 Laptop GPU using
197,000 primitives, one camera, a 1536×810 image, and seed 42. Geometry and direct
appearance came from the existing synthetic benchmark's `make_scene` function.
The SH variant uses degree-three coefficients with equivalent DC colors and zero
higher coefficients. This is not a substitute for the trained models.

The compact results and measurement settings are in
[v011_rtx3060_control.json](../benchmarks/results/v011_rtx3060_control.json).
The matched control measured classic backward at 173.0 → 51.8 ms (3.34x) and
neural backward at 261.0 → 94.5 ms (2.76x). Total render/L1/backward latency was
215.9 → 80.8 ms for classic and 297.8 → 122.4 ms for neural. Compare validation
modes in the report: the neural and SH candidates use `validate=False`; the
classic direct-color comparison keeps validation enabled in both versions.

Each case used five warmups, 12 measured steps, and three profiled steps. All
control processes were pinned to CPU 4. Unpinned runs on this host exposed a
PyTorch 2.8 `ApproximateClock` non-monotonic timestamp error and incomplete CUDA
traces. If that profiler error occurs, pin the process to an available CPU with
`taskset -c <cpu> uv run python benchmarks/benchmark_real_scene.py ...`. The
benchmark records CPU affinity and fails when compositor kernel capture is
missing; it does not substitute zero timings or retry silently.

Correctness includes the original suite and upstream output comparisons, plus
independent SH polynomial/gradient parity at every supported degree, trusted-input
path parity, and full/tail shared-batch coverage. Compute Sanitizer memcheck also
checks multi-scene/camera, partial-tile, packed/dense, and SH camera-sharing stress
cases. Racecheck additionally reports zero hazards for shared-batch/tail and
partial-tile stress cases. The release is prepared locally; publishing the tag, rebuilding a cu129
wheel, updating an external application's pin, and rerunning its original profile
require the corresponding deployment environment and publication approval.
