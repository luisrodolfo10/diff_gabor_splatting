"""Compare loop, fused dense, and fused packed rendering through the public API."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import torch

from gabor_splatting import rasterize_gabors, rasterize_neural_gabors


def make_scene(args: argparse.Namespace, method: str) -> dict[str, torch.Tensor]:
    """Build a deterministic synthetic scene outside the timed region."""
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    b, n, c = args.num_scenes, args.num_primitives, args.num_cameras
    means = torch.randn(b, n, 3, device="cuda", generator=generator) * 0.6
    means[..., 2] = torch.rand(b, n, device="cuda", generator=generator) * 2 + 3
    quaternions = torch.randn(b, n, 4, device="cuda", generator=generator) * 0.1
    quaternions[..., 0] += 1
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    views = torch.eye(4, device="cuda").expand(b, c, 4, 4).clone()
    views[..., 0, 3] = torch.linspace(-0.2, 0.2, c, device="cuda") if c > 1 else 0
    K = torch.tensor(
        [
            [args.image_width * 0.8, 0, (args.image_width - 1) / 2],
            [0, args.image_height * 0.8, (args.image_height - 1) / 2],
            [0, 0, 1],
        ],
        device="cuda",
    )
    result = {
        "means": means,
        "quaternions": quaternions,
        "scales": torch.full((b, n, 2), 0.025, device="cuda"),
        "opacities": torch.full((b, n), 0.3, device="cuda"),
        "view_matrices": views,
        "intrinsics": K.expand(b, c, 3, 3).clone(),
        "backgrounds": torch.full((b, c, 3), 0.05, device="cuda"),
    }
    if method == "classic":
        result.update(
            colors=torch.rand(b, n, 3, device="cuda", generator=generator),
            secondary_colors=torch.rand(b, n, 3, device="cuda", generator=generator),
            frequency_weights=torch.full((b, n, 4), 0.25, device="cuda"),
            frequency_magnitudes=torch.full((b, n, 4), 1.0, device="cuda"),
            phases=torch.zeros(b, n, 4, device="cuda"),
        )
    else:
        for name, shape, bound in [
            ("input_weights", (6, 5), 1 / 5),
            ("input_biases", (6,), 1 / 5),
            ("output_weights", (3, 6), 1 / 30),
            ("output_biases", (3,), 0.1),
        ]:
            result[name] = torch.empty(b, n, *shape, device="cuda").uniform_(
                -bound, bound, generator=generator
            )
    for name, value in result.items():
        value.requires_grad_(name not in {"view_matrices", "intrinsics", "backgrounds"})
    return result


def render_maps(scene, args, method, implementation):
    """Render all scenes/cameras and collect the same maps for every implementation."""
    function = rasterize_gabors if method == "classic" else rasterize_neural_gabors

    def call(values, packed=False):
        """Invoke the renderer and collect all differentiable channels."""
        rgb, alpha, meta = function(
            **values,
            width=args.image_width,
            height=args.image_height,
            packed=packed,
            validate=not args.skip_validation,
        )
        return torch.cat(
            (rgb, alpha, meta["accumulated_depth"], meta["normals"], meta["distortion"]), -1
        )

    if implementation != "single_view_loop":
        return call(scene, packed=implementation == "fused_packed")
    outputs = []
    for scene_id in range(args.num_scenes):
        cameras = []
        for camera_id in range(args.num_cameras):
            values = {
                name: value[scene_id, camera_id : camera_id + 1].contiguous()
                if name in {"view_matrices", "intrinsics", "backgrounds"}
                else value[scene_id].contiguous()
                for name, value in scene.items()
            }
            cameras.append(call(values))
        outputs.append(torch.cat(cameras, 0))
    return torch.stack(outputs)


def check_parity(scene, args, method):
    """Check full images and parameter gradients before reporting any timing."""
    parameters = tuple(t for t in scene.values() if t.requires_grad)
    baseline = render_maps(scene, args, method, "single_view_loop")
    gradients = torch.autograd.grad(baseline.square().mean(), parameters)
    for implementation in ["fused_dense", "fused_packed"]:
        actual = render_maps(scene, args, method, implementation)
        torch.testing.assert_close(actual, baseline, atol=2e-5, rtol=2e-4)
        actual_gradients = torch.autograd.grad(actual.square().mean(), parameters)
        for value, expected in zip(actual_gradients, gradients):
            torch.testing.assert_close(value, expected, atol=2e-5, rtol=2e-3)


def measure(scene, args, method, implementation):
    """Measure graph-enabled forward and backward using CUDA events on equivalent losses."""

    def reset_gradients():
        """Release input gradients outside the measured CUDA interval."""
        for value in scene.values():
            value.grad = None

    for _ in range(args.warmup_iterations):
        reset_gradients()
        render_maps(scene, args, method, implementation).square().mean().backward()
    reset_gradients()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    forward_ms, backward_ms = [], []
    for _ in range(args.measurement_iterations):
        reset_gradients()
        start, middle, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        start.record()
        loss = render_maps(scene, args, method, implementation).square().mean()
        middle.record()
        loss.backward()
        end.record()
        end.synchronize()
        forward_ms.append(start.elapsed_time(middle))
        backward_ms.append(middle.elapsed_time(end))
    forward = sum(forward_ms) / len(forward_ms)
    backward = sum(backward_ms) / len(backward_ms)
    return {
        "implementation": implementation,
        "forward_ms": forward,
        "backward_ms": backward,
        "forward_backward_ms": forward + backward,
        "images_per_second": args.num_scenes * args.num_cameras * 1000 / (forward + backward),
        "peak_allocated_memory_mb": torch.cuda.max_memory_allocated() / 1_000_000,
    }


def profile_kernels(scene, args, method, implementation):
    """Profile CUDA kernels separately from the uninstrumented latency measurements."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        for _ in range(args.profile_iterations):
            for value in scene.values():
                value.grad = None
            render_maps(scene, args, method, implementation).square().mean().backward()
        torch.cuda.synchronize()
    kernels = {}
    for event in profile.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.name] = (
                kernels.get(event.name, 0.0)
                + event.device_time_total / 1000 / args.profile_iterations
            )
    forward = sum(ms for name, ms in kernels.items() if "renderForward<" in name)
    backward = sum(ms for name, ms in kernels.items() if "renderBackward<" in name)
    if not forward or not backward:
        raise RuntimeError("profiler did not capture the forward/backward compositor kernels")
    return {
        "forward_kernel_ms": forward,
        "backward_kernel_ms": backward,
        "other_cuda_kernel_ms": sum(kernels.values()) - forward - backward,
        "kernels_ms": kernels,
    }


def main():
    """Parse a small explicit workload, verify parity, and emit reproducible results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["classic", "neural", "both"], default="both")
    parser.add_argument("--num-primitives", type=int, default=10_000)
    parser.add_argument("--num-scenes", type=int, default=1)
    parser.add_argument("--num-cameras", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--measurement-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=0,
        help="separately profile this many steps per implementation; 0 disables profiling",
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    for name in [
        "num_primitives",
        "num_scenes",
        "num_cameras",
        "image_width",
        "image_height",
        "measurement_iterations",
    ]:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_iterations < 0 or args.profile_iterations < 0:
        parser.error("--warmup-iterations and --profile-iterations must be nonnegative")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    report = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        },
        "workload": {name: value for name, value in vars(args).items() if name != "output_json"},
        "timing": "CUDA events; forward includes autograd graph construction and scalar loss; combined throughput includes backward",
        "profiling": "optional separate CUDA trace; kernel times include instrumentation overhead",
        "memory": "peak total PyTorch-allocated CUDA memory in decimal MB, including inputs and autograd state",
        "results": [],
    }
    for method in ["classic", "neural"] if args.method == "both" else [args.method]:
        scene = make_scene(args, method)
        check_parity(scene, args, method)
        baseline = None
        for implementation in ["single_view_loop", "fused_dense", "fused_packed"]:
            result = measure(scene, args, method, implementation)
            if args.profile_iterations:
                result.update(profile_kernels(scene, args, method, implementation))
            if baseline is None:
                baseline = result["forward_backward_ms"]
            result.update(
                method=method, speedup_vs_single_view_loop=baseline / result["forward_backward_ms"]
            )
            report["results"].append(result)
            print(
                f"{method:7s} {implementation:16s} forward={result['forward_ms']:.3f} ms "
                f"backward={result['backward_ms']:.3f} ms speedup={result['speedup_vs_single_view_loop']:.2f}x",
                flush=True,
            )
        del scene
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
