// Copyright (C) 2023, Inria. See LICENSE.md and NOTICE.md.
// Real SH ordering and constants match the Gabor rasterizers.
#include "rasterize.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>

namespace {
// Evaluate the real SH basis and its direction derivatives through degree three.
template<int Degree>
__device__ void shBasis(float x, float y, float z, float* b, float* dx, float* dy, float* dz) {
    b[0] = 0.28209479177387814f;
    if constexpr (Degree > 0) {
        constexpr float c1 = 0.4886025119029199f;
        b[1] = -c1 * y; dy[1] = -c1;
        b[2] = c1 * z; dz[2] = c1;
        b[3] = -c1 * x; dx[3] = -c1;
    }
    if constexpr (Degree > 1) {
        constexpr float a = 1.0925484305920792f, c = 0.31539156525252005f, d = 0.5462742152960396f;
        b[4] = a*x*y; dx[4] = a*y; dy[4] = a*x;
        b[5] = -a*y*z; dy[5] = -a*z; dz[5] = -a*y;
        b[6] = c*(2*z*z-x*x-y*y); dx[6] = -2*c*x; dy[6] = -2*c*y; dz[6] = 4*c*z;
        b[7] = -a*x*z; dx[7] = -a*z; dz[7] = -a*x;
        b[8] = d*(x*x-y*y); dx[8] = 2*d*x; dy[8] = -2*d*y;
    }
    if constexpr (Degree > 2) {
        constexpr float a=-0.5900435899266435f, b1=2.890611442640554f, c=-0.4570457994644658f;
        constexpr float d=0.3731763325901154f, e=1.445305721320277f;
        b[9] = a*y*(3*x*x-y*y); dx[9] = 6*a*x*y; dy[9] = 3*a*(x*x-y*y);
        b[10] = b1*x*y*z; dx[10] = b1*y*z; dy[10] = b1*x*z; dz[10] = b1*x*y;
        b[11] = c*y*(4*z*z-x*x-y*y); dx[11] = -2*c*x*y;
        dy[11] = c*(4*z*z-x*x-3*y*y); dz[11] = 8*c*y*z;
        b[12] = d*z*(2*z*z-3*x*x-3*y*y); dx[12] = -6*d*x*z;
        dy[12] = -6*d*y*z; dz[12] = d*(6*z*z-3*x*x-3*y*y);
        b[13] = c*x*(4*z*z-x*x-y*y); dx[13] = c*(4*z*z-3*x*x-y*y);
        dy[13] = -2*c*x*y; dz[13] = 8*c*x*z;
        b[14] = e*z*(x*x-y*y); dx[14] = 2*e*x*z; dy[14] = -2*e*y*z; dz[14] = e*(x*x-y*y);
        b[15] = a*x*(x*x-3*y*y); dx[15] = 3*a*(x*x-y*y); dy[15] = -6*a*x*y;
    }
}

// Fuse both SH colors, their clamps, and their coefficient/direction derivatives per pair.
template<int Degree, bool Backward>
__global__ void evaluateSH(int64_t count, int64_t n, int64_t cameras,
    const float* first, const float* second, const float* directions, const float* grad_colors,
    float* colors, float* grad_first, float* grad_second, float* grad_directions) {
    int64_t pair = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= count) return;
    constexpr int K = (Degree + 1) * (Degree + 1);
    int64_t primitive = pair / (cameras * n) * n + pair % n;
    float basis[K] = {}, dx[K] = {}, dy[K] = {}, dz[K] = {};
    shBasis<Degree>(directions[pair*3], directions[pair*3+1], directions[pair*3+2], basis, dx, dy, dz);
    float gd[3] = {};
    #pragma unroll
    for (int color = 0; color < 2; ++color) {
        const float* coefficients = (color == 0 ? first : second) + primitive * K * 3;
        float* gc = color == 0 ? grad_first : grad_second;
        #pragma unroll
        for (int channel = 0; channel < 3; ++channel) {
            float raw = 0.5f;
            #pragma unroll
            for (int k = 0; k < K; ++k) raw += basis[k] * coefficients[k*3+channel];
            if constexpr (Backward) {
                float gradient = raw >= 0 ? grad_colors[pair*6+color*3+channel] : 0;
                #pragma unroll
                for (int k = 0; k < K; ++k) {
                    float value = gradient * basis[k];
                    if (cameras == 1) gc[primitive*K*3+k*3+channel] = value;
                    else if (value != 0) atomicAdd(gc+primitive*K*3+k*3+channel, value);
                    gd[0] += gradient * coefficients[k*3+channel] * dx[k];
                    gd[1] += gradient * coefficients[k*3+channel] * dy[k];
                    gd[2] += gradient * coefficients[k*3+channel] * dz[k];
                }
            } else {
                colors[pair*6+color*3+channel] = fmaxf(raw, 0.0f);
            }
        }
    }
    if constexpr (Backward) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) grad_directions[pair*3+k] = gd[k];
    }
}

// Check SH tensor structure without introducing value checks or device synchronization.
void checkSH(const torch::Tensor& first, const torch::Tensor& second,
             const torch::Tensor& directions, int64_t degree) {
    TORCH_CHECK(degree >= 0 && degree <= 3, "SH degree must be 0 through 3");
    for (const auto& value : {first, second, directions}) {
        TORCH_CHECK(value.is_cuda() && value.scalar_type() == torch::kFloat32
                    && value.is_contiguous() && value.device() == first.device(),
                    "SH tensors must be contiguous float32 on the same CUDA device");
    }
    TORCH_CHECK(first.dim() == 4 && second.sizes() == first.sizes()
                && first.size(2) == (degree+1)*(degree+1) && first.size(3) == 3,
                "SH coefficients must have shape [B, N, K, 3]");
    TORCH_CHECK(directions.dim() == 4 && directions.size(0) == first.size(0)
                && directions.size(2) == first.size(1) && directions.size(3) == 3,
                "SH directions must have shape [B, C, N, 3]");
    TORCH_CHECK(directions.numel()/3 <= INT_MAX, "too many SH pairs");
}
} // namespace

// Evaluate both SH colors in one CUDA launch without a per-polynomial autograd graph.
torch::Tensor evaluateSHForward(torch::Tensor first, torch::Tensor second,
                                torch::Tensor directions, int64_t degree) {
    checkSH(first, second, directions, degree);
    c10::cuda::CUDAGuard guard(first.device());
    auto stream = c10::cuda::getCurrentCUDAStream(first.get_device()).stream();
    auto colors = torch::empty({directions.size(0), directions.size(1), directions.size(2), 6}, first.options());
    int64_t count = directions.numel()/3;
    if (count > 0) {
        #define SH_FORWARD(D) evaluateSH<D, false><<<(count+255)/256,256,0,stream>>>(count, first.size(1), directions.size(1), first.data_ptr<float>(), second.data_ptr<float>(), directions.data_ptr<float>(), nullptr, colors.data_ptr<float>(), nullptr, nullptr, nullptr)
        switch (degree) {
            case 0: SH_FORWARD(0); break;
            case 1: SH_FORWARD(1); break;
            case 2: SH_FORWARD(2); break;
            case 3: SH_FORWARD(3); break;
        }
        #undef SH_FORWARD
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return colors;
}

// Differentiate both colors into shared coefficients and independent camera/primitive directions.
std::vector<torch::Tensor> evaluateSHBackward(torch::Tensor first, torch::Tensor second,
    torch::Tensor directions, torch::Tensor grad_colors, int64_t degree) {
    checkSH(first, second, directions, degree);
    TORCH_CHECK(grad_colors.device() == first.device() && grad_colors.scalar_type() == torch::kFloat32
                && grad_colors.is_contiguous() && grad_colors.sizes() == torch::IntArrayRef({directions.size(0), directions.size(1), directions.size(2), 6}), "invalid SH color gradients");
    c10::cuda::CUDAGuard guard(first.device());
    auto stream = c10::cuda::getCurrentCUDAStream(first.get_device()).stream();
    auto gf = torch::zeros_like(first), gs = torch::zeros_like(second), gd = torch::zeros_like(directions);
    int64_t count = directions.numel()/3;
    if (count > 0) {
        #define SH_BACKWARD(D) evaluateSH<D, true><<<(count+255)/256,256,0,stream>>>(count, first.size(1), directions.size(1), first.data_ptr<float>(), second.data_ptr<float>(), directions.data_ptr<float>(), grad_colors.data_ptr<float>(), nullptr, gf.data_ptr<float>(), gs.data_ptr<float>(), gd.data_ptr<float>())
        switch (degree) {
            case 0: SH_BACKWARD(0); break;
            case 1: SH_BACKWARD(1); break;
            case 2: SH_BACKWARD(2); break;
            case 3: SH_BACKWARD(3); break;
        }
        #undef SH_BACKWARD
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {gf, gs, gd};
}
