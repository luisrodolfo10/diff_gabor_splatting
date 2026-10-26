"""Independent surface/compositing checks and fused-batch regression coverage."""

import pytest
import torch

from gabor_splatting import rasterize_gabors, rasterize_neural_gabors

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


def make_inputs(method, scenes=1, cameras=2, primitives=3):
    """Construct deterministic, overlapping, tilted surfaces in one image tile."""
    generator = torch.Generator(device="cuda").manual_seed(7)
    means = torch.randn(scenes, primitives, 3, generator=generator, device="cuda") * 0.08
    means[..., 2] = torch.linspace(2.4, 3.5, primitives, device="cuda")
    quats = torch.tensor([1.0, 0.08, -0.05, 0.02], device="cuda")
    quats = (quats / quats.norm()).expand(scenes, primitives, 4).clone()
    views = torch.eye(4, device="cuda").expand(scenes, cameras, 4, 4).clone()
    views[..., 0, 3] = torch.linspace(-0.03, 0.04, cameras, device="cuda")
    intrinsics = torch.tensor([[16.0, 0.0, 4.0], [0.0, 17.0, 4.0], [0.0, 0.0, 1.0]], device="cuda")
    values = dict(
        means=means,
        quaternions=quats,
        scales=torch.full((scenes, primitives, 2), 0.37, device="cuda"),
        opacities=torch.full((scenes, primitives), 0.32, device="cuda"),
        view_matrices=views,
        intrinsics=intrinsics.expand(scenes, cameras, 3, 3).clone(),
        backgrounds=torch.full((scenes, cameras, 3), 0.13, device="cuda"),
    )
    if method == "classic":
        values.update(
            colors=torch.rand(scenes, primitives, 3, generator=generator, device="cuda"),
            secondary_colors=torch.rand(scenes, primitives, 3, generator=generator, device="cuda"),
            frequency_weights=torch.full((scenes, primitives, 4), 0.25, device="cuda"),
            frequency_magnitudes=torch.full((scenes, primitives, 4), 0.4, device="cuda"),
            phases=torch.rand(scenes, primitives, 4, generator=generator, device="cuda"),
        )
    else:
        for name, shape in [
            ("input_weights", (6, 5)),
            ("input_biases", (6,)),
            ("output_weights", (3, 6)),
            ("output_biases", (3,)),
        ]:
            values[name] = (
                torch.randn(scenes, primitives, *shape, generator=generator, device="cuda") * 0.08
            )
    return {name: value.requires_grad_() for name, value in values.items()}


def renderer(method):
    """Select one of the two public renderers for the same validation workload."""
    return rasterize_gabors if method == "classic" else rasterize_neural_gabors


def maps_from_result(result):
    """Collect every differentiable output into one tensor for weighted checks."""
    rgb, alpha, meta = result
    return torch.cat(
        (rgb, alpha, meta["accumulated_depth"], meta["normals"], meta["distortion"]), -1
    )


def reference(method, values):
    """Evaluate small visible scenes using float64 PyTorch ray solves and compositing."""
    v = {name: value.double() for name, value in values.items()}
    y, x = torch.meshgrid(
        torch.arange(9, device="cuda", dtype=torch.float64),
        torch.arange(9, device="cuda", dtype=torch.float64),
        indexing="ij",
    )
    pixels = torch.stack((x, y), -1)
    images = []
    for scene in range(v["means"].shape[0]):
        for camera in range(v["view_matrices"].shape[1]):
            view, K = v["view_matrices"][scene, camera], v["intrinsics"][scene, camera]
            transmittance = torch.ones_like(x)
            rgb = torch.zeros_like(pixels[..., :1]).expand(9, 9, 3).clone()
            depth, distortion = torch.zeros_like(x), torch.zeros_like(x)
            normal_map = torch.zeros_like(rgb)
            m1, m2 = torch.zeros_like(x), torch.zeros_like(x)
            stopped = torch.zeros_like(x, dtype=torch.bool)
            for primitive in range(v["means"].shape[1]):
                q = v["quaternions"][scene, primitive]
                q = q / q.norm()
                w, axis = q[0], q[1:]
                a, b, c = axis.unbind()
                zero = w * 0
                skew = torch.stack((zero, -c, b, c, zero, -a, -b, a, zero)).reshape(3, 3)
                R = (w * w - axis.square().sum()) * torch.eye(3, device="cuda")
                R = R + 2 * axis[:, None] * axis[None, :] + 2 * w * skew
                center = view[:3, :3] @ v["means"][scene, primitive] + view[:3, 3]
                tangent = view[:3, :3] @ (R[:, :2] * v["scales"][scene, primitive])
                T = K @ torch.cat((tangent, center[:, None]), -1)
                k = x[..., None] * T[2] - T[0]
                plane_y = y[..., None] * T[2] - T[1]
                A = torch.stack((k[..., :2], plane_y[..., :2]), -2)
                rhs = -torch.stack((k[..., 2], plane_y[..., 2]), -1)
                s = torch.linalg.solve(A, rhs)
                f = (
                    T.new_tensor([9.0, 9.0, -1.0])
                    / (T.new_tensor([9.0, 9.0, -1.0]) * T[2].square()).sum()
                )
                xy = (f * T[:2] * T[2]).sum(-1)
                rho3 = s.square().sum(-1)
                rho2 = 2 * (xy - pixels).square().sum(-1)
                z = (s * T[2, :2]).sum(-1) + T[2, 2]
                if method == "classic":
                    z = torch.where(rho3 <= rho2, z, T[2, 2])
                alpha = (
                    v["opacities"][scene, primitive] * torch.exp(-0.5 * torch.minimum(rho3, rho2))
                ).clamp_max(0.99)
                stopped = stopped | ((alpha >= 1 / 255) & (transmittance * (1 - alpha) < 1e-4))
                accepted = (alpha >= 1 / 255) & (z >= 0.2) & ~stopped
                alpha = torch.where(accepted, alpha, 0)
                if method == "classic":
                    angles = torch.arange(4, device="cuda", dtype=torch.float64) * torch.pi / 4
                    directions = torch.stack((angles.cos(), angles.sin()), -1)
                    phase = (
                        2
                        * torch.pi
                        * v["frequency_magnitudes"][scene, primitive]
                        * (s @ directions.T)
                    )
                    wave = (phase + v["phases"][scene, primitive]).cos()
                    weights = v["frequency_weights"][scene, primitive]
                    h1, h2 = (weights * (1 + wave) / 2).sum(-1), (weights * (1 - wave) / 2).sum(-1)
                    color = (
                        h1[..., None] * v["colors"][scene, primitive]
                        + h2[..., None] * v["secondary_colors"][scene, primitive]
                    )
                else:
                    camera_center = -view[:3, :3].T @ view[:3, 3]
                    direction = v["means"][scene, primitive] - camera_center
                    direction = direction / direction.norm()
                    inputs = torch.cat((s / 3, direction.expand(9, 9, 3)), -1)
                    hidden = torch.sin(
                        30
                        * (
                            inputs @ v["input_weights"][scene, primitive].T
                            + v["input_biases"][scene, primitive]
                        )
                    )
                    color = torch.sigmoid(
                        hidden @ v["output_weights"][scene, primitive].T
                        + v["output_biases"][scene, primitive]
                    )
                normal = view[:3, :3] @ R[:, 2]
                normal = normal * torch.where(-(normal * center).sum() > 0, 1.0, -1.0)
                weight = transmittance * alpha
                m = 100 / 99.8 * (1 - 0.2 / z)
                distortion = distortion + weight * (m * m * (1 - transmittance) + m2 - 2 * m * m1)
                m1, m2 = m1 + weight * m, m2 + weight * m * m
                rgb = rgb + weight[..., None] * color
                depth = depth + weight * z
                normal_map = normal_map + weight[..., None] * normal
                transmittance = transmittance * (1 - alpha)
            rgb = rgb + transmittance[..., None] * v["backgrounds"][scene, camera]
            images.append(
                torch.cat(
                    (
                        rgb,
                        (1 - transmittance)[..., None],
                        depth[..., None],
                        normal_map,
                        distortion[..., None],
                    ),
                    -1,
                )
            )
    return torch.stack(images).reshape(v["means"].shape[0], -1, 9, 9, 9)


@pytest.mark.parametrize("method", ["classic", "neural"])
@pytest.mark.parametrize("packed", [False, True])
def test_reference_outputs_and_gradients(method, packed):
    """Compare every image output and every input derivative with an independent reference."""
    values = make_inputs(method)
    actual = maps_from_result(renderer(method)(**values, width=9, height=9, packed=packed))
    expected = reference(method, values)
    torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=2e-5)
    weights = torch.linspace(-0.4, 0.7, actual.numel(), device="cuda").reshape(actual.shape)
    actual_grads = torch.autograd.grad(
        (actual * weights).sum(), tuple(values.values()), retain_graph=True
    )
    reference_grads = torch.autograd.grad((expected * weights).sum(), tuple(values.values()))
    for name, actual_grad, expected_grad in zip(values, actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-3, rtol=8e-4, msg=name)


@pytest.mark.parametrize("method", ["classic", "neural"])
@pytest.mark.parametrize("packed", [False, True])
def test_scene_camera_batch_matches_loop(method, packed):
    """Check that multiple scenes and cameras share the same images and summed derivatives."""
    values = make_inputs(method, scenes=2, cameras=3)
    result = maps_from_result(renderer(method)(**values, width=9, height=9, packed=packed))
    repeated = []
    for scene in range(2):
        views = []
        for camera in range(3):
            subset = {
                name: value[scene, camera : camera + 1].contiguous()
                if name in {"view_matrices", "intrinsics", "backgrounds"}
                else value[scene].contiguous()
                for name, value in values.items()
            }
            views.append(maps_from_result(renderer(method)(**subset, width=9, height=9)))
        repeated.append(torch.cat(views))
    repeated = torch.stack(repeated)
    torch.testing.assert_close(result, repeated)
    actual = torch.autograd.grad(result.square().sum(), tuple(values.values()), retain_graph=True)
    expected = torch.autograd.grad(repeated.square().sum(), tuple(values.values()))
    for name, a, e in zip(values, actual, expected):
        torch.testing.assert_close(a, e, atol=4e-3, rtol=1e-3, msg=name)


@pytest.mark.parametrize("method", ["classic", "neural"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_empty_and_invisible_scenes(method, packed, empty):
    """Return background-only images and finite zero primitive gradients when nothing is visible."""
    values = make_inputs(method, primitives=0 if empty else 3)
    if not empty:
        with torch.no_grad():
            values["means"][..., 2] = -2
    rgb, alpha, meta = renderer(method)(**values, width=9, height=9, packed=packed)
    torch.testing.assert_close(rgb, values["backgrounds"][:, :, None, None, :].expand_as(rgb))
    assert alpha.count_nonzero() == 0
    maps_from_result((rgb, alpha, meta)).sum().backward()
    for name, value in values.items():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
        if name != "backgrounds":
            assert value.grad.count_nonzero() == 0, name


@pytest.mark.parametrize("method", ["classic", "neural"])
def test_leading_dimensions_and_stream(method):
    """Preserve multiple scene dimensions and obey the caller's non-default CUDA stream."""
    with torch.cuda.stream(torch.cuda.Stream()):
        values = make_inputs(method, scenes=4)
        values = {name: value.reshape(2, 2, *value.shape[1:]) for name, value in values.items()}
        rgb, alpha, _ = renderer(method)(**values, width=9, height=9, packed=True)
        (rgb.sum() + alpha.sum()).backward()
        assert rgb.shape == (2, 2, 2, 9, 9, 3)
    torch.cuda.synchronize()


@pytest.mark.parametrize("method", ["classic", "neural"])
def test_invalid_contracts(method):
    """Fail explicitly for unsupported dtypes, shapes, layouts, and numerical parameters."""
    original = make_inputs(method)
    for name, replacement in [
        ("means", original["means"].double()),
        ("scales", -original["scales"]),
        ("quaternions", torch.zeros_like(original["quaternions"])),
        ("opacities", torch.full_like(original["opacities"], 2)),
        ("intrinsics", original["intrinsics"].transpose(-1, -2)),
    ]:
        values = dict(original, **{name: replacement})
        with pytest.raises(ValueError):
            renderer(method)(**values, width=9, height=9)
    with pytest.raises(ValueError):
        renderer(method)(**original, width=0, height=9)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_classic_sh_batches(degree):
    """Exercise SH color gradients and camera-specific appearance in packed and dense modes."""
    values = make_inputs("classic", scenes=2)
    for name in ["colors", "secondary_colors"]:
        values[name] = torch.full(
            (2, 3, (degree + 1) ** 2, 3), 0.1, device="cuda", requires_grad=True
        )
    dense = maps_from_result(rasterize_gabors(**values, width=9, height=9, sh_degree=degree))
    packed = maps_from_result(
        rasterize_gabors(**values, width=9, height=9, sh_degree=degree, packed=True)
    )
    torch.testing.assert_close(dense, packed)
    dense.sum().backward()
    assert values["colors"].grad.abs().sum() > 0
    assert values["secondary_colors"].grad.abs().sum() > 0


@pytest.mark.parametrize("method", ["classic", "neural"])
@pytest.mark.parametrize("opacity", [0.0, 1.0])
def test_opacity_cutoff_and_saturation(method, opacity):
    """Check zero opacity, clamped alpha, and early termination against the reference."""
    values = make_inputs(method, cameras=1, primitives=7)
    with torch.no_grad():
        values["opacities"].fill_(0.85 if opacity == 1.0 else opacity)
        values["opacities"][..., 0] = opacity
        values["quaternions"].zero_()
        values["quaternions"][..., 0] = 1
        values["means"][..., :2] = 0
    actual = maps_from_result(renderer(method)(**values, width=9, height=9))
    expected = reference(method, values)
    torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=3e-5)
    actual_grads = torch.autograd.grad(actual.sum(), tuple(values.values()))
    expected_grads = torch.autograd.grad(expected.sum(), tuple(values.values()))
    for name, a, e in zip(values, actual_grads, expected_grads):
        torch.testing.assert_close(a, e, atol=2e-3, rtol=2e-3, msg=name)


@pytest.mark.parametrize("method", ["classic", "neural"])
def test_tiny_surfaces_use_lowpass_filter(method):
    """Validate filtered opacity and appearance when the surface is smaller than a pixel."""
    values = make_inputs(method, cameras=1, primitives=1)
    with torch.no_grad():
        values["scales"].fill_(0.02)
        values["quaternions"].zero_()
        values["quaternions"][..., 0] = 1
    actual = maps_from_result(renderer(method)(**values, width=9, height=9))
    expected = reference(method, values)
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-4)
    actual_grads = torch.autograd.grad(actual.sum(), tuple(values.values()))
    expected_grads = torch.autograd.grad(expected.sum(), tuple(values.values()))
    for name, a, e in zip(values, actual_grads, expected_grads):
        torch.testing.assert_close(a, e, atol=5e-3, rtol=2e-3, msg=name)


@pytest.mark.parametrize("degree", [0, 3])
def test_empty_sh_scene(degree):
    """Keep SH empty-scene shapes valid without inferring a dimension from zero elements."""
    values = make_inputs("classic", primitives=0)
    for name in ["colors", "secondary_colors"]:
        values[name] = torch.empty((1, 0, (degree + 1) ** 2, 3), device="cuda", requires_grad=True)
    result = rasterize_gabors(**values, width=9, height=9, sh_degree=degree, packed=True)
    maps_from_result(result).sum().backward()
    assert values["colors"].grad.shape == values["colors"].shape


@pytest.mark.parametrize("method", ["classic", "neural"])
def test_parameter_finite_differences(method):
    """Check representative geometry and appearance derivatives with central differences."""
    values = make_inputs(method, cameras=1, primitives=1)
    names = ["means", "scales", "opacities", "phases" if method == "classic" else "input_biases"]

    def objective():
        """Use a weighted scalar image loss away from discrete visibility boundaries."""
        maps = maps_from_result(renderer(method)(**values, width=9, height=9))
        weights = torch.linspace(-0.2, 0.4, maps.numel(), device="cuda").reshape(maps.shape)
        return (maps.double() * weights).sum()

    gradients = torch.autograd.grad(objective(), [values[name] for name in names])
    for name, gradient in zip(names, gradients):
        parameter = values[name]
        index = (0,) * parameter.ndim
        step = 5e-4
        original = parameter[index].item()
        with torch.no_grad():
            parameter[index] = original + step
        plus = objective().item()
        with torch.no_grad():
            parameter[index] = original - step
        minus = objective().item()
        with torch.no_grad():
            parameter[index] = original
        torch.testing.assert_close(
            gradient[index].double(),
            gradient.new_tensor((plus - minus) / (2 * step)).double(),
            atol=3e-3,
            rtol=8e-3,
            msg=name,
        )


@pytest.mark.parametrize("method", ["classic", "neural"])
def test_float32_inputs_ignore_global_default_dtype(method):
    """Preserve explicit input dtype even when the application's default is float64."""
    values = make_inputs(method, cameras=1, primitives=1)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        rgb, alpha, _ = renderer(method)(**values, width=9, height=9)
        (rgb.sum() + alpha.sum()).backward()
        assert rgb.dtype == torch.float32 and alpha.dtype == torch.float32
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("method", ["classic", "neural"])
@pytest.mark.parametrize("case", ["edge_on", "degenerate_conic"])
def test_degenerate_surfaces_have_finite_zero_gradients(method, case):
    """Cull exactly edge-on and zero-denominator conics without contaminating autograd."""
    values = make_inputs(method, cameras=1, primitives=1)
    with torch.no_grad():
        values["view_matrices"].copy_(torch.eye(4, device="cuda"))
        values["means"][0, 0] = values["means"].new_tensor([0.0, 0.0, 3.0])
        values["quaternions"].fill_(0.5)
        if case == "degenerate_conic":
            values["scales"].fill_(1.0)
            values["means"][..., 0] = 0.1
    rgb, alpha, meta = renderer(method)(**values, width=9, height=9, packed=True)
    assert alpha.count_nonzero() == 0
    maps_from_result((rgb, alpha, meta)).sum().backward()
    for name, value in values.items():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
        if name != "backgrounds":
            assert value.grad.count_nonzero() == 0, name
