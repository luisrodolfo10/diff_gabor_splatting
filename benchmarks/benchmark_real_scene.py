"""Measure one exported real tile, including CUDA forward/backward kernel timings.

Input is a torch.save bundle: {"method": "classic" | "neural", "inputs": public
renderer keyword arguments including width/height, "target": channels-last RGB,
"trainable": a list of tensor argument names}. Export activated renderer inputs,
not an application-specific model checkpoint. Loading and device copies are untimed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from gabor_splatting import rasterize_gabors, rasterize_neural_gabors


def load_tile(path: Path, device: torch.device):
    """Load explicit renderer arguments, a reference RGB tile, and trainable parameter names."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle["method"] not in {"classic", "neural"}:
        raise ValueError("method must be classic or neural")
    inputs = {
        name: value.to(device).detach().contiguous() if isinstance(value, torch.Tensor) else value
        for name, value in bundle["inputs"].items()
    }
    if "validate" in inputs:
        raise ValueError("choose validation with --skip-validation, not inside the input bundle")
    parameters = []
    for name in bundle["trainable"]:
        inputs[name].requires_grad_(True)
        parameters.append(inputs[name])
    if not parameters:
        raise ValueError("trainable must name at least one renderer input")
    target = bundle["target"].to(device).contiguous()
    if target.dtype != torch.float32 or target.shape[-3:] != (inputs["height"], inputs["width"], 3):
        raise ValueError("target must be float32 channels-last RGB matching width/height")
    if target.numel() != inputs["height"] * inputs["width"] * 3:
        raise ValueError("this benchmark measures exactly one scene and one camera")
    return bundle["method"], inputs, parameters, target


def benchmark_tile(method, inputs, parameters, target, warmups, iterations, profile_iterations):
    """Measure render/L1/backward without optimizer updates and separately profile CUDA kernels."""
    function = rasterize_gabors if method == "classic" else rasterize_neural_gabors

    def step():
        """Clear gradients and differentiate the L1 tile reconstruction objective."""
        for parameter in parameters:
            parameter.grad = None
        rgb, _, _ = function(**inputs)
        if rgb.numel() != target.numel():
            raise ValueError("renderer output must describe the same single RGB tile as target")
        loss = (rgb.reshape_as(target) - target).abs().mean()
        loss.backward()
        return loss

    for _ in range(warmups):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        loss = step()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000 / iterations
    step_ms = start.elapsed_time(end) / iterations
    peak_mb = torch.cuda.max_memory_allocated() / 1_000_000
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        for _ in range(profile_iterations):
            step()
        torch.cuda.synchronize()
    kernels = {}
    for event in profile.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.name] = (
                kernels.get(event.name, 0.0) + event.device_time_total / 1000 / profile_iterations
            )
    sh_ms = sum(ms for name, ms in kernels.items() if "evaluateSH<" in name)
    forward_ms = sum(ms for name, ms in kernels.items() if "renderForward<" in name)
    backward_ms = sum(ms for name, ms in kernels.items() if "renderBackward<" in name)
    if not forward_ms or not backward_ms:
        raise RuntimeError(
            f"profiler did not capture the compositor kernels: {len(kernels)} CUDA names, "
            f"matches={[name for name in kernels if 'render' in name]}"
        )
    profiled_ms = sum(kernels.values())
    return {
        "method": method,
        "num_primitives": inputs["means"].shape[-2],
        "width": inputs["width"],
        "height": inputs["height"],
        "validate": inputs.get("validate", True),
        "packed": inputs.get("packed", False),
        "step_ms": step_ms,
        "wall_ms_per_step": wall_ms,
        "sh_kernel_ms": sh_ms,
        "forward_kernel_ms": forward_ms,
        "backward_kernel_ms": backward_ms,
        "other_cuda_kernel_ms": profiled_ms - forward_ms - backward_ms,
        "profiled_cuda_kernel_ms": profiled_ms,
        "kernels_ms": kernels,
        "peak_allocated_memory_mb": peak_mb,
        "final_loss": loss.item(),
    }


def main():
    """Profile an exported real scene without inventing checkpoint or training conventions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tile", type=Path, help="torch.save bundle of actual tile inputs")
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--measurement-iterations", type=int, default=30)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--max-step-ms", type=float, help="fail if measured step latency exceeds this budget"
    )
    args = parser.parse_args()
    if args.warmup_iterations < 0 or min(args.measurement_iterations, args.profile_iterations) < 1:
        parser.error("warmups must be nonnegative and measurement/profile iterations positive")
    if args.max_step_ms is not None and not 0 < args.max_step_ms < float("inf"):
        parser.error("--max-step-ms must be finite and positive")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("a CUDA device is required")
    torch.cuda.set_device(device)
    method, inputs, parameters, target = load_tile(args.tile, device)
    if args.skip_validation:
        inputs["validate"] = False
    result = benchmark_tile(
        method,
        inputs,
        parameters,
        target,
        args.warmup_iterations,
        args.measurement_iterations,
        args.profile_iterations,
    )
    revision = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
    report = {
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "revision": revision.stdout.strip(),
        },
        "tile": str(args.tile),
        "warmups": args.warmup_iterations,
        "iterations": args.measurement_iterations,
        "profile_iterations": args.profile_iterations,
        "objective": "render + RGB L1 loss + backward; no optimizer or densification",
        "timing": "step_ms uses CUDA events after warmups; kernel times use a separate profiler trace; other_cuda_kernel_ms includes other CUDA activities, not CPU work",
        "result": result,
    }
    if args.max_step_ms is not None:
        report["max_step_ms"] = args.max_step_ms
        report["meets_step_budget"] = result["step_ms"] <= args.max_step_ms
    print(json.dumps(report, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    if args.max_step_ms is not None and result["step_ms"] > args.max_step_ms:
        raise SystemExit(
            f"step {result['step_ms']:.3f} ms exceeds budget {args.max_step_ms:.3f} ms"
        )


if __name__ == "__main__":
    main()
