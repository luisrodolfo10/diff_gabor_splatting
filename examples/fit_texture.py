"""Fit a small stripe texture using classic or neural Gabor appearance parameters."""

import argparse
import math
from pathlib import Path

import torch

from gabor_splatting import rasterize_gabors, rasterize_neural_gabors


def fit_texture(method: str, steps: int):
    """Fit one surface and return the target, initial render, and final render on CPU."""
    torch.manual_seed(3)
    geometry = {
        "means": torch.tensor([[0.0, 0.0, 3.0]], device="cuda"),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        "scales": torch.tensor([[0.7, 0.7]], device="cuda"),
        "opacities": torch.tensor([0.9], device="cuda"),
        "view_matrices": torch.eye(4, device="cuda")[None],
        "intrinsics": torch.tensor(
            [[[40.0, 0.0, 15.5], [0.0, 40.0, 15.5], [0.0, 0.0, 1.0]]], device="cuda"
        ),
        "width": 32,
        "height": 32,
    }
    x = torch.arange(32, device="cuda")[None, None, :, None].expand(1, 32, 32, 3)
    target = (0.5 + 0.4 * torch.cos(x * 0.7)).contiguous()
    if method == "classic":
        values = {
            "colors": torch.tensor([[0.6, 0.4, 0.5]], device="cuda", requires_grad=True),
            "secondary_colors": torch.tensor([[0.3, 0.5, 0.4]], device="cuda", requires_grad=True),
            "frequency_weights": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
            "frequency_magnitudes": torch.full((1, 4), 1.0, device="cuda", requires_grad=True),
            "phases": torch.zeros(1, 4, device="cuda", requires_grad=True),
        }
        function = rasterize_gabors
    else:
        values = {
            "input_weights": torch.empty(1, 6, 5, device="cuda")
            .uniform_(-1 / 5, 1 / 5)
            .requires_grad_(),
            "input_biases": torch.zeros(1, 6, device="cuda", requires_grad=True),
            "output_weights": torch.empty(1, 3, 6, device="cuda")
            .uniform_(-math.sqrt(6 / 6) / 30, math.sqrt(6 / 6) / 30)
            .requires_grad_(),
            "output_biases": torch.zeros(1, 3, device="cuda", requires_grad=True),
        }
        function = rasterize_neural_gabors
    optimizer = torch.optim.Adam([v for v in values.values() if v.requires_grad], lr=0.01)
    with torch.no_grad():
        initial, alpha, _ = function(**geometry, **values)
        target_image = (target * alpha).cpu()
        initial_image = initial.cpu()
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        rgb, alpha, _ = function(**geometry, **values)
        loss = ((rgb - target * alpha.detach()).square()).mean()
        loss.backward()
        optimizer.step()
        if method == "classic":
            with torch.no_grad():
                values["frequency_magnitudes"].clamp_min_(0)
        losses.append(loss.item())
        if step == 0 or step == steps - 1:
            print(f"{method} step={step + 1} loss={losses[-1]:.6f}")
    if not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError("optimization produced non-finite losses")
    with torch.no_grad():
        final, _, _ = function(**geometry, **values)
    return target_image, initial_image, final.cpu()


def save_comparison(results: dict, steps: int, output: Path) -> None:
    """Save labeled target/before/after panels from actual renderer outputs using Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    panel, left, gap, margin = 224, 184, 20, 24
    width = left + 3 * panel + 2 * gap + margin
    height = 150 + len(results) * (panel + 64)
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=32)
    label = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=17)
    draw.text((margin, 22), "Learning a local stripe texture", font=title, fill="#0f172a")
    draw.text(
        (margin, 66),
        f"One surface | 32 x 32 pixels | {steps} optimization steps",
        font=small,
        fill="#475569",
    )
    for column, heading in enumerate(["Target", "Initial render", "Fitted render"]):
        draw.text((left + column * (panel + gap), 112), heading, font=label, fill="#334155")
    for row, (method, panels) in enumerate(results.items()):
        y = 150 + row * (panel + 64)
        draw.text(
            (margin, y + 78),
            "Classic Gabor" if method == "classic" else "Neural Gabor",
            font=label,
            fill="#0f172a",
        )
        draw.text(
            (margin, y + 112),
            "4 frequencies" if method == "classic" else "5-6-3 SIREN",
            font=small,
            fill="#64748b",
        )
        for column, tensor in enumerate(panels):
            pixels = (tensor[0].clamp(0, 1) * 255).round().to(torch.uint8).contiguous()
            raster = Image.frombytes("RGB", (32, 32), bytes(pixels.flatten().tolist()))
            raster = raster.resize((panel, panel), Image.Resampling.NEAREST)
            image.paste(raster, (left + column * (panel + gap), y))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Saved {output}")


def main():
    """Run one or both fits and optionally save a reproducible README comparison image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["classic", "neural", "both"], default="classic")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, help="save target/initial/fitted panels as a PNG (requires Pillow)"
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    methods = ["classic", "neural"] if args.method == "both" else [args.method]
    results = {method: fit_texture(method, args.steps) for method in methods}
    if args.output:
        save_comparison(results, args.steps, args.output)


if __name__ == "__main__":
    main()
