"""Compare fused SH values and derivatives with the independent polynomial reference."""

import pytest
import torch
from gabor_splatting.rendering import SHColors

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


def evaluate_sh_reference(
    degree: int, coeffs: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    """Evaluate real spherical harmonics up to degree three."""

    if degree < 0 or degree > 3:
        raise ValueError("sh_degree must be between 0 and 3")
    if coeffs.shape[-2] < (degree + 1) ** 2:
        raise ValueError("colors does not contain enough SH coefficients")
    x, y, z = directions.unbind(-1)
    result = 0.28209479177387814 * coeffs[..., 0, :]
    if degree > 0:
        result = (
            result
            - 0.4886025119029199 * y[..., None] * coeffs[..., 1, :]
            + 0.4886025119029199 * z[..., None] * coeffs[..., 2, :]
            - 0.4886025119029199 * x[..., None] * coeffs[..., 3, :]
        )
    if degree > 1:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        c2 = (
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396,
        )
        result = result + c2[0] * xy[..., None] * coeffs[..., 4, :]
        result = result + c2[1] * yz[..., None] * coeffs[..., 5, :]
        result = result + c2[2] * (2 * zz - xx - yy)[..., None] * coeffs[..., 6, :]
        result = result + c2[3] * xz[..., None] * coeffs[..., 7, :]
        result = result + c2[4] * (xx - yy)[..., None] * coeffs[..., 8, :]
    if degree > 2:
        c3 = (
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435,
        )
        result = result + c3[0] * y[..., None] * (3 * x * x - y * y)[..., None] * coeffs[..., 9, :]
        result = result + c3[1] * (x * y * z)[..., None] * coeffs[..., 10, :]
        result = result + (
            c3[2] * y[..., None] * (4 * z * z - x * x - y * y)[..., None] * coeffs[..., 11, :]
        )
        result = result + (
            c3[3]
            * z[..., None]
            * (2 * z * z - 3 * x * x - 3 * y * y)[..., None]
            * coeffs[..., 12, :]
        )
        result = result + (
            c3[4] * x[..., None] * (4 * z * z - x * x - y * y)[..., None] * coeffs[..., 13, :]
        )
        result = result + c3[5] * z[..., None] * (x * x - y * y)[..., None] * coeffs[..., 14, :]
        result = result + c3[6] * x[..., None] * (x * x - 3 * y * y)[..., None] * coeffs[..., 15, :]
    return (result + 0.5).clamp_min(0.0)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("cameras", [1, 3])
def test_fused_sh_values_and_derivatives(degree, cameras):
    """Preserve clamp, camera sharing, and direction/parameter gradients at all SH degrees."""
    torch.manual_seed(31)
    first = torch.randn(2, 7, (degree + 1) ** 2, 3, device="cuda", requires_grad=True)
    second = torch.randn_like(first, requires_grad=True)
    directions = torch.randn(2, cameras, 7, 3, device="cuda")
    directions = (directions / directions.norm(dim=-1, keepdim=True)).requires_grad_()
    actual = SHColors.apply(first, second, directions, degree)
    expected = torch.cat(
        [
            evaluate_sh_reference(degree, c[:, None], directions).expand(2, cameras, 7, 3)
            for c in (first, second)
        ],
        -1,
    )
    expected = expected + directions.sum() * 0
    weights = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * weights).sum(), (first, second, directions))
    expected_grads = torch.autograd.grad((expected * weights).sum(), (first, second, directions))
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, atol=3e-5, rtol=2e-4)
