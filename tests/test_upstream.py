"""Optional parity checks against separately built pinned upstream rasterizers."""

import importlib
import importlib.util

import pytest
import torch
from test_rendering import make_inputs, renderer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@pytest.mark.parametrize("method", ["classic", "neural"])
def test_pinned_upstream_outputs(method):
    """Compare RGB and surface maps across multiple tiles with upstream conventions."""
    module_name = (
        "diff_gabor_rasterization" if method == "classic" else "diff_neural_gabor_rasterization"
    )
    if importlib.util.find_spec(module_name) is None:
        pytest.skip(f"optional upstream package {module_name} is not installed")
    upstream = importlib.import_module(module_name)
    values = make_inputs(method, cameras=1, primitives=10)
    width, height = 35, 33
    with torch.no_grad():
        values["means"][..., 0] = torch.linspace(-1.1, 1.1, 10, device="cuda")
        values["means"][..., 1] = torch.linspace(0.6, -0.6, 10, device="cuda")
        values["intrinsics"][..., 0, 0] = 28
        values["intrinsics"][..., 1, 1] = 27
        values["intrinsics"][..., 0, 2] = (width - 1) / 2
        values["intrinsics"][..., 1, 2] = (height - 1) / 2
    rgb, alpha, meta = renderer(method)(**values, width=width, height=height)
    K = values["intrinsics"][0, 0]
    view = values["view_matrices"][0, 0]
    projection = torch.zeros(4, 4, device="cuda")
    projection[0, 0] = 2 * K[0, 0] / width
    projection[1, 1] = 2 * K[1, 1] / height
    projection[0, 2] = (2 * K[0, 2] - (width - 1)) / width
    projection[1, 2] = (2 * K[1, 2] - (height - 1)) / height
    projection[2, 2] = 100 / 99.8
    projection[2, 3] = -100 * 0.2 / 99.8
    projection[3, 2] = 1
    settings_type = (
        upstream.GaborRasterizationSettings
        if method == "classic"
        else upstream.GaussianRasterizationSettings
    )
    rasterizer_type = (
        upstream.GaborRasterizer if method == "classic" else upstream.GaussianRasterizer
    )
    settings = settings_type(
        image_height=height,
        image_width=width,
        tanfovx=width / (2 * K[0, 0].item()),
        tanfovy=height / (2 * K[1, 1].item()),
        bg=values["backgrounds"][0, 0],
        scale_modifier=1.0,
        viewmatrix=view.T.contiguous(),
        projmatrix=(projection @ view).T.contiguous(),
        sh_degree=0,
        campos=(-view[:3, :3].T @ view[:3, 3]).contiguous(),
        prefiltered=False,
        debug=False,
    )
    arguments = dict(
        means3D=values["means"][0],
        means2D=torch.zeros_like(values["means"][0]),
        opacities=values["opacities"][0, :, None].contiguous(),
        scales=values["scales"][0],
        rotations=values["quaternions"][0],
    )
    if method == "classic":
        arguments.update(
            colors_precomp=values["colors"][0],
            colors_precomp2=values["secondary_colors"][0],
            weights_for_frequency_angle=values["frequency_weights"][0],
            frequency_lengths=values["frequency_magnitudes"][0],
            phases=values["phases"][0],
        )
    else:
        arguments.update(
            W1=values["input_weights"][0],
            b1=values["input_biases"][0],
            W2=values["output_weights"][0],
            b2=values["output_biases"][0],
        )
    expected_rgb, radii, maps = rasterizer_type(settings)(**arguments)
    torch.testing.assert_close(rgb[0, 0], expected_rgb.permute(1, 2, 0), atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(alpha[0, 0, ..., 0], maps[1], atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(
        meta["accumulated_depth"][0, 0, ..., 0], maps[0], atol=3e-5, rtol=3e-4
    )
    torch.testing.assert_close(
        meta["normals"][0, 0], maps[2:5].permute(1, 2, 0), atol=3e-5, rtol=3e-4
    )
    torch.testing.assert_close(meta["distortion"][0, 0, ..., 0], maps[6], atol=3e-6, rtol=3e-3)
    torch.testing.assert_close(meta["radii"][0, 0], radii)
