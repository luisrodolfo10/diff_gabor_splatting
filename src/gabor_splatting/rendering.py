"""Public CUDA surface renderers and explicit tensor contracts."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from . import _C


class Composite(torch.autograd.Function):
    """Bridge the fused native compositor to first-order PyTorch autograd."""

    @staticmethod
    def forward(
        ctx: Any,
        geometry,
        appearance,
        opacities,
        primitive_ids,
        image_ids,
        radii,
        backgrounds,
        width,
        height,
        neural,
        near_plane,
        far_plane,
    ):
        """Render maps and retain the sorted contributor state for backward."""
        maps, points, ranges, state, contributors = _C.rasterize_forward(
            geometry,
            appearance,
            opacities,
            primitive_ids,
            image_ids,
            radii,
            backgrounds,
            width,
            height,
            neural,
            near_plane,
            far_plane,
        )
        ctx.save_for_backward(
            geometry,
            appearance,
            opacities,
            primitive_ids,
            backgrounds,
            points,
            ranges,
            state,
            contributors,
        )
        ctx.options = (width, height, neural, near_plane, far_plane)
        return maps

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_maps):
        """Differentiate all continuous image maps; sorting and culling stay discrete."""
        gg, ga, go, gb = _C.rasterize_backward(
            *ctx.saved_tensors,
            grad_maps.contiguous(),
            *ctx.options,
        )
        return gg, ga, go, None, None, None, gb, None, None, None, None, None


def check_tensor(
    name: str, value: torch.Tensor, shape: tuple[int, ...], device: torch.device
) -> None:
    """Require a contiguous float32 CUDA tensor with an exact shape and device."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device != device or value.dtype != torch.float32 or not value.is_cuda:
        raise ValueError(f"{name} must be float32 on {device}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@torch.no_grad()
def check_values(
    tensors, quaternions, scales, opacities, view_matrices, intrinsics, appearance, neural
):
    """Validate numerical contracts in one synchronization and identify violated inputs."""
    checks = [
        (f"{name} must contain only finite values", torch.isfinite(t).all())
        for name, (t, _) in tensors.items()
    ]
    checks += [
        ("scales must be strictly positive", (scales > 0).all()),
        ("opacities must lie in [0, 1]", ((opacities >= 0) & (opacities <= 1)).all()),
        (
            "quaternions must be unit wxyz vectors",
            torch.isclose(
                quaternions.square().sum(-1), quaternions.new_tensor(1), atol=1e-4, rtol=1e-4
            ).all(),
        ),
        (
            "view_matrices must have last row [0, 0, 0, 1]",
            torch.isclose(view_matrices[..., 3, :], view_matrices.new_tensor([0, 0, 0, 1])).all(),
        ),
        (
            "intrinsics must have last row [0, 0, 1]",
            torch.isclose(intrinsics[..., 2, :], intrinsics.new_tensor([0, 0, 1])).all(),
        ),
        (
            "intrinsics fx and fy must be positive",
            ((intrinsics[..., 0, 0] > 0) & (intrinsics[..., 1, 1] > 0)).all(),
        ),
    ]
    rotation = view_matrices[..., :3, :3]
    checks += [
        (
            "view_matrices rotations must be orthonormal",
            torch.isclose(
                rotation.transpose(-1, -2) @ rotation,
                torch.eye(3, device=rotation.device, dtype=rotation.dtype),
                atol=1e-4,
                rtol=1e-4,
            ).all(),
        ),
        (
            "view_matrices rotations must have determinant +1",
            torch.isclose(
                torch.linalg.det(rotation), rotation.new_tensor(1), atol=1e-4, rtol=1e-4
            ).all(),
        ),
    ]
    if not neural:
        checks += [
            ("frequency_weights must be nonnegative", (appearance[2] >= 0).all()),
            ("frequency_magnitudes must be nonnegative", (appearance[3] >= 0).all()),
            (
                "frequency_weights must sum to one",
                torch.isclose(
                    appearance[2].sum(-1), appearance[2].new_tensor(1), atol=1e-4, rtol=1e-4
                ).all(),
            ),
        ]
    statuses = torch.stack([valid for _, valid in checks])
    if not bool(statuses.all()):
        violations = [
            message for (message, _), valid in zip(checks, statuses.tolist()) if not valid
        ]
        raise ValueError("; ".join(violations))


def quaternion_matrix(q: torch.Tensor) -> torch.Tensor:
    """Normalize wxyz quaternions and convert them to rotation matrices."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def project_surfaces(
    means, quaternions, scales, view_matrices, intrinsics, width, height, near_plane, far_plane
):
    """Project tangent planes, bounding centers, facing normals, and world view directions."""
    rotation = quaternion_matrix(quaternions)
    tangent = rotation[..., :2] * scales.unsqueeze(-2)
    camera_rotation = view_matrices[..., :3, :3]
    camera_means = torch.einsum("bcij,bnj->bcni", camera_rotation, means)
    camera_means = camera_means + view_matrices[..., None, :3, 3]
    camera_tangent = torch.einsum("bcij,bnjk->bcnik", camera_rotation, tangent)
    plane = torch.cat((camera_tangent, camera_means.unsqueeze(-1)), dim=-1)
    transform = torch.einsum("bcij,bcnjk->bcnik", intrinsics, plane)
    # Bound the three-sigma disk in perspective, following the upstream conic.
    t = transform.new_tensor([9.0, 9.0, -1.0])
    denominator = (t * transform[..., 2, :].square()).sum(-1)
    valid_conic = denominator != 0
    safe_denominator = torch.where(valid_conic, denominator, torch.ones_like(denominator))
    f = t / safe_denominator.unsqueeze(-1)
    xy = (f.unsqueeze(-2) * transform[..., :2, :] * transform[..., 2:3, :]).sum(-1)
    extent_square = xy.square() - (f.unsqueeze(-2) * transform[..., :2, :].square()).sum(-1)
    extent = extent_square.clamp_min(1e-4).sqrt()
    normals = torch.einsum("bcij,bnj->bcni", camera_rotation, rotation[..., 2])
    facing = -(camera_means * normals).sum(-1)
    normals = normals * torch.where(
        facing > 0, torch.ones_like(facing), -torch.ones_like(facing)
    ).unsqueeze(-1)
    camera_centers = -torch.einsum("bcji,bcj->bci", camera_rotation, view_matrices[..., :3, 3])
    direction = means[:, None] - camera_centers[:, :, None]
    directions = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(
        1e-12
    )
    with torch.no_grad():
        radius = torch.maximum(extent.amax(-1), extent.new_tensor(3 * 0.707106)).ceil()
        tw, th = (width + 15) // 16, (height + 15) // 16
        lower = torch.trunc((xy - radius.unsqueeze(-1)) / 16).clamp_min(0)
        upper = torch.trunc((xy + radius.unsqueeze(-1) + 15) / 16).clamp_min(0)
        visible = valid_conic & (facing != 0) & (camera_means[..., 2] > near_plane)
        visible &= camera_means[..., 2] < far_plane
        visible &= torch.isfinite(radius) & (radius < 2**30) & torch.isfinite(xy).all(-1)
        visible &= upper[..., 0].clamp_max(tw) > lower[..., 0].clamp_max(tw)
        visible &= upper[..., 1].clamp_max(th) > lower[..., 1].clamp_max(th)
        radii = torch.where(visible, radius, 0).to(torch.int32)
    return transform, xy, normals, directions, radii


class SHColors(torch.autograd.Function):
    """Fuse the two classic SH colors and their first-order CUDA derivatives."""

    @staticmethod
    def forward(ctx: Any, first, second, directions, degree):
        """Evaluate both coefficient sets using a shared direction and SH basis."""
        ctx.save_for_backward(first, second, directions)
        ctx.degree = degree
        return _C.evaluate_sh_forward(first, second, directions, degree)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_colors):
        """Accumulate SH coefficient gradients across cameras and differentiate directions."""
        gradients = _C.evaluate_sh_backward(
            *ctx.saved_tensors, grad_colors.contiguous(), ctx.degree
        )
        return *gradients, None


def render(
    means,
    quaternions,
    scales,
    opacities,
    appearance,
    view_matrices,
    intrinsics,
    width,
    height,
    backgrounds,
    packed,
    near_plane,
    far_plane,
    neural,
    sh_degree,
    validate,
):
    """Validate inputs, project scene/camera pairs, and call one image-aware compositor."""
    if not isinstance(means, torch.Tensor) or means.ndim < 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape [..., N, 3]")
    if not means.is_cuda or means.dtype != torch.float32:
        raise ValueError("means must be a CUDA float32 tensor")
    if (
        type(width) is not int
        or type(height) is not int
        or not (0 < width < 2**30 and 0 < height < 2**30)
    ):
        raise ValueError("width and height must be positive integers smaller than 2**30")
    if (
        not math.isfinite(near_plane)
        or not math.isfinite(far_plane)
        or not 0 < near_plane < far_plane
    ):
        raise ValueError("clipping planes must be finite and satisfy 0 < near_plane < far_plane")
    if type(packed) is not bool:
        raise TypeError("packed must be bool")
    batch = tuple(means.shape[:-2])
    n = means.shape[-2]
    if not isinstance(view_matrices, torch.Tensor) or view_matrices.ndim != len(batch) + 3:
        raise ValueError("view_matrices must have shape [..., C, 4, 4]")
    cameras = view_matrices.shape[-3]
    scenes = math.prod(batch)
    if scenes < 1 or cameras < 1:
        raise ValueError("scene and camera dimensions must be nonempty")
    device = means.device
    tensors = {
        "means": (means, batch + (n, 3)),
        "quaternions": (quaternions, batch + (n, 4)),
        "scales": (scales, batch + (n, 2)),
        "opacities": (opacities, batch + (n,)),
        "view_matrices": (view_matrices, batch + (cameras, 4, 4)),
        "intrinsics": (intrinsics, batch + (cameras, 3, 3)),
    }
    if backgrounds is None:
        backgrounds = means.new_zeros(batch + (cameras, 3))
    tensors["backgrounds"] = backgrounds, batch + (cameras, 3)
    if neural:
        appearance_shapes = [(6, 5), (6,), (3, 6), (3,)]
        appearance_names = ["input_weights", "input_biases", "output_weights", "output_biases"]
    else:
        if sh_degree is not None and (type(sh_degree) is not int or not 0 <= sh_degree <= 3):
            raise ValueError("sh_degree must be None or an integer from 0 to 3")
        color_shape = (3,) if sh_degree is None else ((sh_degree + 1) ** 2, 3)
        appearance_shapes = [color_shape, color_shape, (4,), (4,), (4,)]
        appearance_names = [
            "colors",
            "secondary_colors",
            "frequency_weights",
            "frequency_magnitudes",
            "phases",
        ]
    for name, tensor, shape in zip(appearance_names, appearance, appearance_shapes):
        tensors[name] = tensor, batch + (n,) + shape
    for name, (tensor, shape) in tensors.items():
        check_tensor(name, tensor, shape, device)
    if type(validate) is not bool:
        raise TypeError("validate must be bool")
    if validate:
        check_values(
            tensors, quaternions, scales, opacities, view_matrices, intrinsics, appearance, neural
        )
    transform, xy, normals, directions, radii = project_surfaces(
        means.reshape(scenes, n, 3),
        quaternions.reshape(scenes, n, 4),
        scales.reshape(scenes, n, 2),
        view_matrices.reshape(scenes, cameras, 4, 4),
        intrinsics.reshape(scenes, cameras, 3, 3),
        width,
        height,
        near_plane,
        far_plane,
    )
    if sh_degree is not None:
        colors = SHColors.apply(
            appearance[0].reshape(scenes, n, (sh_degree + 1) ** 2, 3),
            appearance[1].reshape(scenes, n, (sh_degree + 1) ** 2, 3),
            directions,
            sh_degree,
        )
        rest = [t.reshape(scenes, 1, n, 4).expand(scenes, cameras, n, 4) for t in appearance[2:]]
        parameters = torch.cat((colors, *rest), dim=-1).reshape(-1, 18).contiguous()
        opacity_parameters = (
            opacities.reshape(scenes, 1, n).expand(scenes, cameras, n).reshape(-1).contiguous()
        )
        primitive_ids = torch.arange(scenes * cameras * n, device=device)
    else:
        parameter_count = 57 if neural else 18
        parameters = torch.cat(
            [
                t.reshape(scenes, n, math.prod(shape))
                for t, shape in zip(appearance, appearance_shapes)
            ],
            dim=-1,
        )
        parameters = parameters.reshape(scenes * n, parameter_count).contiguous()
        opacity_parameters = opacities.reshape(scenes * n)
        primitive_ids = torch.arange(scenes * n, device=device).reshape(scenes, 1, n)
        primitive_ids = primitive_ids.expand(scenes, cameras, n).reshape(-1).contiguous()
    image_ids = torch.arange(scenes * cameras, device=device).reshape(scenes, cameras, 1)
    image_ids = image_ids.expand(scenes, cameras, n).reshape(-1).contiguous()
    selected = torch.arange(scenes * cameras * n, device=device)
    if packed:
        selected = selected[radii.reshape(-1) > 0]
    projected_means = xy.reshape(-1, 2)[selected]
    geometry = torch.cat(
        (
            transform.reshape(-1, 9)[selected],
            projected_means,
            normals.reshape(-1, 3)[selected],
            directions.reshape(-1, 3)[selected],
        ),
        dim=-1,
    ).contiguous()
    maps = Composite.apply(
        geometry,
        parameters,
        opacity_parameters,
        primitive_ids[selected],
        image_ids[selected],
        radii.reshape(-1)[selected].contiguous(),
        backgrounds.reshape(-1, 3),
        width,
        height,
        neural,
        near_plane,
        far_plane,
    ).reshape(batch + (cameras, height, width, 9))
    meta = {
        "radii": radii.reshape(-1)[selected] if packed else radii.reshape(batch + (cameras, n)),
        "projected_means": projected_means,
        "scene_ids": selected // (cameras * max(n, 1)),
        "camera_ids": selected // max(n, 1) % cameras,
        "primitive_ids": selected % max(n, 1),
        "packed": packed,
        "accumulated_depth": maps[..., 4:5],
        "normals": maps[..., 5:8],
        "distortion": maps[..., 8:9],
    }
    return maps[..., :3], maps[..., 3:4], meta


def rasterize_gabors(
    means: torch.Tensor,
    quaternions: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    secondary_colors: torch.Tensor,
    frequency_weights: torch.Tensor,
    frequency_magnitudes: torch.Tensor,
    phases: torch.Tensor,
    view_matrices: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    *,
    backgrounds: torch.Tensor | None = None,
    packed: bool = False,
    near_plane: float = 0.2,
    far_plane: float = 100.0,
    sh_degree: int | None = None,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Render four-frequency Gabor surfaces over arbitrary scene and camera batches.

    Geometry is [..., N, 3] means, [..., N, 4] unit wxyz rotations,
    [..., N, 2] positive tangent scales, and [..., N] activated opacities.
    Colors are [..., N, 3], or [..., N, (degree+1)**2, 3] SH coefficients.
    Weights, nonnegative frequency magnitudes (cycles per scaled local unit), and
    phases (radians) are [..., N, 4]. Nonnegative weights must sum to one.
    Cameras are [..., C, 4, 4] rigid world-to-camera transforms and [..., C, 3, 3]
    pinhole intrinsics. All tensors must be contiguous CUDA float32.
    Returns RGB [..., C, H, W, 3], alpha [..., C, H, W, 1], and named metadata.
    validate=False skips numerical value checks for trusted inputs; shape,
    dtype, device, and contiguity checks always remain enabled.
    See docs/contracts.md for pixel conventions, map definitions, and gradients.
    """
    return render(
        means,
        quaternions,
        scales,
        opacities,
        (colors, secondary_colors, frequency_weights, frequency_magnitudes, phases),
        view_matrices,
        intrinsics,
        width,
        height,
        backgrounds,
        packed,
        near_plane,
        far_plane,
        False,
        sh_degree,
        validate,
    )


def rasterize_neural_gabors(
    means: torch.Tensor,
    quaternions: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    input_weights: torch.Tensor,
    input_biases: torch.Tensor,
    output_weights: torch.Tensor,
    output_biases: torch.Tensor,
    view_matrices: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    *,
    backgrounds: torch.Tensor | None = None,
    packed: bool = False,
    near_plane: float = 0.2,
    far_plane: float = 100.0,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Render per-surface 5→6→3 SIRENs with sine frequency 30 and sigmoid RGB.

    Geometry and camera contracts match rasterize_gabors. Per-primitive input
    weights/biases have shapes [..., N, 6, 5] and [..., N, 6]; output weights/
    biases have shapes [..., N, 3, 6] and [..., N, 3]. Input is local (u/3, v/3)
    followed by the world-space unit direction from camera to primitive center.
    All tensors must be contiguous CUDA float32. Returns channels-last RGB,
    alpha, and named metadata, with first-order parameter and camera gradients.
    validate=False skips numerical value checks for trusted inputs; shape,
    dtype, device, and contiguity checks always remain enabled.
    """
    return render(
        means,
        quaternions,
        scales,
        opacities,
        (input_weights, input_biases, output_weights, output_biases),
        view_matrices,
        intrinsics,
        width,
        height,
        backgrounds,
        packed,
        near_plane,
        far_plane,
        True,
        None,
        validate,
    )
