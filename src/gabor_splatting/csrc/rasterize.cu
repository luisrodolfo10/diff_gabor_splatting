// Copyright (C) 2023, Inria. See LICENSE.md and NOTICE.md.
// Surface equations follow the pinned Gabor rasterizers; batching uses image/tile keys.
#include "rasterize.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <climits>
#include <cmath>

namespace {
namespace cg = cooperative_groups;
constexpr int Tile = 16;
constexpr int BlockSize = Tile * Tile;
constexpr int BatchSize = 64;
constexpr int GeometrySize = 17;
constexpr int MapSize = 9;
constexpr float Pi = 3.14159265358979323846f;

// Compute the tile rectangle using the same integer bounds as the surface rasterizers.
__device__ int4 tileBounds(const float* geometry, int radius, int tw, int th) {
    return make_int4(
        max(0, min(tw, int((geometry[9] - radius) / Tile))),
        max(0, min(th, int((geometry[10] - radius) / Tile))),
        max(0, min(tw, int((geometry[9] + radius + Tile - 1) / Tile))),
        max(0, min(th, int((geometry[10] + radius + Tile - 1) / Tile))));
}

// Count tile intersections per projected primitive before the allocation scan.
__global__ void countTiles(int count, int tw, int th, const float* geometry,
                           const int* radii, int64_t* counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (radii[i] <= 0) { counts[i] = 0; return; }
    int4 r = tileBounds(geometry + i * GeometrySize, radii[i], tw, th);
    counts[i] = int64_t(r.z - r.x) * (r.w - r.y);
}

// Emit image/tile/depth keys and projected-primitive IDs for one global sort.
__global__ void emitTiles(int count, int tw, int th, const float* geometry,
    const int* radii, const int64_t* image_ids, const int64_t* offsets,
    uint64_t* keys, int* values) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || radii[i] <= 0) return;
    const float* g = geometry + i * GeometrySize;
    int4 r = tileBounds(g, radii[i], tw, th);
    int64_t out = i == 0 ? 0 : offsets[i - 1];
    for (int y = r.y; y < r.w; ++y) for (int x = r.x; x < r.z; ++x) {
        uint64_t tile = uint64_t(image_ids[i]) * tw * th + y * tw + x;
        keys[out] = (tile << 32) | __float_as_uint(g[8]);
        values[out++] = i;
    }
}

// Identify sorted runs belonging to each image tile.
__global__ void findRanges(int count, const uint64_t* keys, int2* ranges) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t tile = keys[i] >> 32;
    if (i == 0) ranges[tile].x = 0;
    if (i > 0 && tile != uint32_t(keys[i - 1] >> 32)) {
        ranges[uint32_t(keys[i - 1] >> 32)].y = i;
        ranges[tile].x = i;
    }
    if (i == count - 1) ranges[tile].y = count;
}

struct Sample {
    float k[3], l[3], p[3], s[2], depth, exponential, alpha;
    bool surface;
};

// Form a cross product for the surface intersection and its adjoint.
__device__ void crossProduct(const float* a, const float* b, float* c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

// Intersect a pixel ray with a surface and evaluate its filtered opacity.
__device__ bool sampleSurface(const float* g, float opacity, int x, int y,
                              bool neural, float near_plane, Sample& v) {
    for (int k = 0; k < 3; ++k) {
        v.k[k] = x * g[6 + k] - g[k];
        v.l[k] = y * g[6 + k] - g[3 + k];
    }
    crossProduct(v.k, v.l, v.p);
    if (v.p[2] == 0) return false;
    v.s[0] = v.p[0] / v.p[2];
    v.s[1] = v.p[1] / v.p[2];
    float rho3 = v.s[0] * v.s[0] + v.s[1] * v.s[1];
    float dx = g[9] - x, dy = g[10] - y;
    float rho2 = 2 * (dx * dx + dy * dy);
    v.surface = rho3 <= rho2;
    v.depth = (neural || v.surface) ? v.s[0] * g[6] + v.s[1] * g[7] + g[8] : g[8];
    if (v.depth < near_plane || !isfinite(v.depth)) return false;
    v.exponential = expf(-0.5f * fminf(rho3, rho2));
    v.alpha = fminf(0.99f, opacity * v.exponential);
    return v.alpha >= 1.0f / 255.0f;
}

// Evaluate either the four-frequency color mixture or the fixed per-primitive SIREN.
template<bool Neural>
__device__ void appearanceForward(const float* a, const float* g, const Sample& v,
                                  float* color, float* hidden, float* preactivation) {
    if constexpr (Neural) {
        float input[5] = {v.s[0] / 3, v.s[1] / 3, g[14], g[15], g[16]};
        #pragma unroll
        for (int h = 0; h < 6; ++h) {
            float z = a[30 + h];
            #pragma unroll
            for (int k = 0; k < 5; ++k) z += a[h * 5 + k] * input[k];
            preactivation[h] = z;
            hidden[h] = sinf(30 * z);
        }
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            float z = a[54 + c];
            #pragma unroll
            for (int h = 0; h < 6; ++h) z += a[36 + c * 6 + h] * hidden[h];
            color[c] = 1 / (1 + expf(-z));
        }
    } else {
        float h1 = 0, h2 = 0;
        #pragma unroll
        for (int f = 0; f < 4; ++f) {
            float dot = cosf(Pi * f / 4) * v.s[0] + sinf(Pi * f / 4) * v.s[1];
            float wave = cosf(2 * Pi * a[10 + f] * dot + a[14 + f]);
            h1 += a[6 + f] * 0.5f * (1 + wave);
            h2 += a[6 + f] * 0.5f * (1 - wave);
        }
        #pragma unroll
        for (int c = 0; c < 3; ++c) color[c] = h1 * a[c] + h2 * a[3 + c];
    }
}

// Combine one primitive's pixel derivatives before issuing a single atomic per warp.
__device__ __forceinline__ void warpAtomicAdd(cg::thread_block_tile<32> warp, float* target, float value) {
    float sum = cg::reduce(warp, value, cg::plus<float>());
    if (warp.thread_rank() == 0 && sum != 0.0f) atomicAdd(target, sum);
}

// Differentiate appearance into primitive parameters, local coordinates, and view direction.
template<bool Neural>
__device__ void appearanceBackward(const float* a, const float* g, const Sample& v,
    const float* color, const float* hidden, const float* preactivation,
    const float* gc, float* ga, float* gg, float* gs, cg::thread_block_tile<32> warp) {
    if constexpr (Neural) {
        float input[5] = {v.s[0] / 3, v.s[1] / 3, g[14], g[15], g[16]};
        float gh[6] = {}, gi[5] = {};
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            float gz = gc[c] * color[c] * (1 - color[c]);
            warpAtomicAdd(warp, ga + 54 + c, gz);
            #pragma unroll
            for (int h = 0; h < 6; ++h) {
                warpAtomicAdd(warp, ga + 36 + c * 6 + h, gz * hidden[h]);
                gh[h] += gz * a[36 + c * 6 + h];
            }
        }
        #pragma unroll
        for (int h = 0; h < 6; ++h) {
            float gz = gh[h] * 30 * cosf(30 * preactivation[h]);
            warpAtomicAdd(warp, ga + 30 + h, gz);
            #pragma unroll
            for (int k = 0; k < 5; ++k) {
                warpAtomicAdd(warp, ga + h * 5 + k, gz * input[k]);
                gi[k] += gz * a[h * 5 + k];
            }
        }
        gs[0] += gi[0] / 3; gs[1] += gi[1] / 3;
        for (int k = 0; k < 3; ++k) gg[14 + k] += gi[k + 2];
    } else {
        #pragma unroll
        for (int f = 0; f < 4; ++f) {
            float dx = cosf(Pi * f / 4), dy = sinf(Pi * f / 4);
            float dot = dx * v.s[0] + dy * v.s[1];
            float angle = 2 * Pi * a[10 + f] * dot + a[14 + f];
            float wave = cosf(angle), gw = 0, gt = 0;
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                ga[c] += gc[c] * a[6 + f] * 0.5f * (1 + wave);
                ga[3 + c] += gc[c] * a[6 + f] * 0.5f * (1 - wave);
                gw += gc[c] * 0.5f * ((1 + wave) * a[c] + (1 - wave) * a[3 + c]);
                gt += gc[c] * a[6 + f] * 0.5f * sinf(angle) * (a[3 + c] - a[c]);
            }
            ga[6 + f] += gw;
            ga[10 + f] += gt * 2 * Pi * dot;
            ga[14 + f] += gt;
            gs[0] += gt * 2 * Pi * a[10 + f] * dx;
            gs[1] += gt * 2 * Pi * a[10 + f] * dy;
        }
    }
}

// Shared primitive data used by every pixel in a tile batch.
template<int AppearanceSize>
struct PrimitiveBatch {
    int ids[BatchSize];
    int64_t primitives[BatchSize];
    float geometry[BatchSize * GeometrySize];
    float appearance[BatchSize * AppearanceSize];
    float opacities[BatchSize];
};

// Cooperatively gather contiguous primitive records; all tile threads must participate.
template<int AppearanceSize>
__device__ void loadBatch(PrimitiveBatch<AppearanceSize>& batch, int start, int stride,
    int count, int rank, const int* point_list, const int64_t* primitive_ids,
    const float* geometry, const float* appearance, const float* opacities) {
    __syncthreads();
    if (rank < count) {
        int id = point_list[start + stride * rank];
        batch.ids[rank] = id;
        batch.primitives[rank] = primitive_ids[id];
        batch.opacities[rank] = opacities[primitive_ids[id]];
    }
    __syncthreads();
    for (int i = rank; i < count * GeometrySize; i += BlockSize)
        batch.geometry[i] = geometry[int64_t(batch.ids[i / GeometrySize]) * GeometrySize + i % GeometrySize];
    for (int i = rank; i < count * AppearanceSize; i += BlockSize)
        batch.appearance[i] = appearance[batch.primitives[i / AppearanceSize] * AppearanceSize + i % AppearanceSize];
    __syncthreads();
}

// Composite shared primitive batches, keeping per-pixel accumulation in registers.
template<bool Neural>
__global__ __launch_bounds__(BlockSize) void renderForward(int width, int height, int tw, const float* geometry,
    const float* appearance, const float* opacities, const int64_t* primitive_ids,
    const float* backgrounds, const int* point_list, const int2* ranges,
    float near_plane, float far_plane, float* maps, float* state, int* contributors) {
    constexpr int AppearanceSize = Neural ? 57 : 18;
    __shared__ PrimitiveBatch<AppearanceSize> batch;
    int th = (height + Tile - 1) / Tile, tile = blockIdx.x;
    int image = tile / (tw * th), local = tile % (tw * th);
    int x = (local % tw) * Tile + threadIdx.x, y = (local / tw) * Tile + threadIdx.y;
    int rank = threadIdx.y * Tile + threadIdx.x;
    bool inside = x < width && y < height, done = !inside;
    int64_t pixel = (int64_t(image) * height + y) * width + x;
    float out[MapSize] = {}, T = 1, m1 = 0, m2 = 0;
    int last = 0;
    int2 range = ranges[tile];
    for (int start = range.x; start < range.y; start += BatchSize) {
        if (__syncthreads_count(done) == BlockSize) break;
        int count = min(BatchSize, range.y - start);
        loadBatch(batch, start, 1, count, rank, point_list, primitive_ids, geometry, appearance, opacities);
        for (int j = 0; j < count && !done; ++j) {
            const float* g = batch.geometry + j * GeometrySize;
            Sample v;
            if (!sampleSurface(g, batch.opacities[j], x, y, Neural, near_plane, v)) continue;
            float next = T * (1 - v.alpha);
            if (next < 0.0001f) { done = true; break; }
            float color[3], hidden[6], preactivation[6];
            appearanceForward<Neural>(batch.appearance + j * AppearanceSize, g, v, color, hidden, preactivation);
            float w = T * v.alpha;
            float m = far_plane / (far_plane - near_plane) * (1 - near_plane / v.depth);
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                out[c] += w * color[c];
                out[5 + c] += w * g[11 + c];
            }
            out[4] += w * v.depth;
            out[8] += w * (m * m * (1 - T) + m2 - 2 * m * m1);
            m1 += w * m; m2 += w * m * m;
            T = next;
            last = start + j - range.x + 1;
        }
    }
    if (inside) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) out[c] += T * backgrounds[image * 3 + c];
        out[3] = 1 - T;
        #pragma unroll
        for (int c = 0; c < MapSize; ++c) maps[pixel * MapSize + c] = out[c];
        state[pixel * 3] = T; state[pixel * 3 + 1] = m1; state[pixel * 3 + 2] = m2;
        contributors[pixel] = last;
    }
}

// Replay shared reverse batches and reduce all primitive gradients within each warp.
template<bool Neural>
__global__ __launch_bounds__(BlockSize) void renderBackward(int width, int height, int tw, const float* geometry,
    const float* appearance, const float* opacities, const int64_t* primitive_ids,
    const float* backgrounds, const int* point_list, const int2* ranges,
    const float* state, const int* contributors, const float* grad_maps,
    float near_plane, float far_plane, float* grad_geometry, float* grad_appearance,
    float* grad_opacities, float* grad_backgrounds) {
    constexpr int AppearanceSize = Neural ? 57 : 18;
    __shared__ PrimitiveBatch<AppearanceSize> batch;
    using BlockMax = cub::BlockReduce<int, Tile, cub::BLOCK_REDUCE_WARP_REDUCTIONS, Tile>;
    __shared__ BlockMax::TempStorage max_storage;
    __shared__ int block_last;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int th = (height + Tile - 1) / Tile, tile = blockIdx.x;
    int image = tile / (tw * th), local = tile % (tw * th);
    int x = (local % tw) * Tile + threadIdx.x, y = (local / tw) * Tile + threadIdx.y;
    int rank = threadIdx.y * Tile + threadIdx.x;
    bool inside = x < width && y < height;
    int64_t pixel = (int64_t(image) * height + y) * width + x;
    int last = inside ? contributors[pixel] : 0;
    int maximum = BlockMax(max_storage).Reduce(last, cub::Max());
    if (rank == 0) block_last = maximum;
    float go[MapSize] = {};
    if (inside) {
        #pragma unroll
        for (int c = 0; c < MapSize; ++c) go[c] = grad_maps[pixel * MapSize + c];
    }
    float final_t = inside ? state[pixel * 3] : 0, T = final_t;
    float total_a = 1 - final_t;
    float m1 = inside ? state[pixel * 3 + 1] : 0, m2 = inside ? state[pixel * 3 + 2] : 0;
    float background_dot = 0, suffix = 0;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        background_dot += go[c] * backgrounds[image * 3 + c];
        warpAtomicAdd(warp, grad_backgrounds + image * 3 + c, final_t * go[c]);
    }
    __syncthreads();
    int2 range = ranges[tile];
    for (int start = block_last; start > 0; start -= BatchSize) {
        int count = min(BatchSize, start);
        loadBatch(batch, range.x + start - 1, -1, count, rank, point_list, primitive_ids,
                  geometry, appearance, opacities);
        for (int j = 0; j < count; ++j) {
            const float* g = batch.geometry + j * GeometrySize;
            const float* a = batch.appearance + j * AppearanceSize;
            Sample v{};
            bool active = inside && start - j <= last;
            if (active) active = sampleSurface(g, batch.opacities[j], x, y, Neural, near_plane, v);
            if (!warp.ballot(active)) continue;
            if (!active) v = Sample{};
            float gg[GeometrySize] = {}, ga[Neural ? 1 : AppearanceSize] = {}, gopacity = 0;
            float color[3] = {}, hidden[6] = {}, preactivation[6] = {}, gc[3] = {}, gs[2] = {};
            float w = 0, m = 0, galpha = 0;
            float mscale = far_plane / (far_plane - near_plane);
            if (active) {
                T /= 1 - v.alpha;
                w = T * v.alpha;
                appearanceForward<Neural>(a, g, v, color, hidden, preactivation);
                m = mscale * (1 - near_plane / v.depth);
                float gw = go[4] * v.depth + go[8] * (m * m * total_a + m2 - 2 * m * m1);
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    gw += go[c] * color[c] + go[5 + c] * g[11 + c];
                    gc[c] = w * go[c];
                    gg[11 + c] += w * go[5 + c];
                }
                galpha = T * gw - suffix / (1 - v.alpha)
                             + final_t * (go[3] - background_dot) / (1 - v.alpha);
                suffix += w * gw;
            }
            int id = batch.ids[j];
            int64_t primitive = batch.primitives[j];
            appearanceBackward<Neural>(a, g, v, color, hidden, preactivation, gc,
                Neural ? grad_appearance + primitive * AppearanceSize : ga, gg, gs, warp);
            if (active) {
                float gz = w * go[4] + go[8] * 2 * w * (m * total_a - m1)
                           * mscale * near_plane / (v.depth * v.depth);
                if (Neural || v.surface) {
                    gs[0] += gz * g[6]; gs[1] += gz * g[7];
                    gg[6] += gz * v.s[0]; gg[7] += gz * v.s[1];
                }
                gg[8] += gz;
                float grho = 0;
                if (batch.opacities[j] * v.exponential < 0.99f) {
                    gopacity = galpha * v.exponential;
                    grho = -0.5f * galpha * batch.opacities[j] * v.exponential;
                }
                if (v.surface) {
                    gs[0] += 2 * v.s[0] * grho; gs[1] += 2 * v.s[1] * grho;
                } else {
                    gg[9] += 4 * (g[9] - x) * grho;
                    gg[10] += 4 * (g[10] - y) * grho;
                }
                float gp[3] = {gs[0] / v.p[2], gs[1] / v.p[2],
                                -(gs[0] * v.s[0] + gs[1] * v.s[1]) / v.p[2]};
                float gk[3], gl[3];
                crossProduct(v.l, gp, gk); crossProduct(gp, v.k, gl);
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    gg[k] -= gk[k]; gg[3 + k] -= gl[k];
                    gg[6 + k] += x * gk[k] + y * gl[k];
                }
            }

            #pragma unroll
            for (int k = 0; k < GeometrySize; ++k)
                warpAtomicAdd(warp, grad_geometry + int64_t(id) * GeometrySize + k, gg[k]);
            if constexpr (!Neural) {
                #pragma unroll
                for (int k = 0; k < AppearanceSize; ++k)
                    warpAtomicAdd(warp, grad_appearance + primitive * AppearanceSize + k, ga[k]);
            }
            warpAtomicAdd(warp, grad_opacities + primitive, gopacity);
        }
    }
}

// Validate every native floating-point buffer before taking a typed device pointer.
void checkFloat(const torch::Tensor& value, const torch::Tensor& geometry) {
    TORCH_CHECK(value.is_cuda() && value.scalar_type() == torch::kFloat32,
                "native rendering requires CUDA float32 tensors");
    TORCH_CHECK(value.device() == geometry.device() && value.is_contiguous(),
                "native buffers must be contiguous on the geometry device");
}

// Check the common geometry, appearance, IDs, and image background contracts.
void checkBuffers(torch::Tensor geometry, torch::Tensor appearance, torch::Tensor opacities,
                  torch::Tensor primitive_ids, torch::Tensor backgrounds, bool neural) {
    checkFloat(geometry, geometry); checkFloat(appearance, geometry);
    checkFloat(opacities, geometry); checkFloat(backgrounds, geometry);
    TORCH_CHECK(geometry.dim() == 2 && geometry.size(1) == GeometrySize, "invalid geometry shape");
    TORCH_CHECK(geometry.size(0) <= INT_MAX, "too many projected primitives");
    TORCH_CHECK(appearance.dim() == 2 && appearance.size(1) == (neural ? 57 : 18), "invalid appearance shape");
    TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == appearance.size(0), "invalid opacity shape");
    TORCH_CHECK(backgrounds.dim() == 2 && backgrounds.size(1) == 3 && backgrounds.size(0) > 0, "invalid backgrounds");
    TORCH_CHECK(primitive_ids.device() == geometry.device() && primitive_ids.is_contiguous()
                && primitive_ids.scalar_type() == torch::kInt64
                && primitive_ids.dim() == 1 && primitive_ids.size(0) == geometry.size(0), "invalid primitive IDs");
}
} // namespace

// Allocate sorted image-tile runs and dispatch the selected appearance compositor.
std::vector<torch::Tensor> rasterizeForward(
    torch::Tensor geometry, torch::Tensor appearance, torch::Tensor opacities,
    torch::Tensor primitive_ids, torch::Tensor image_ids, torch::Tensor radii,
    torch::Tensor backgrounds, int64_t width, int64_t height, bool neural,
    double near_plane, double far_plane) {
    checkBuffers(geometry, appearance, opacities, primitive_ids, backgrounds, neural);
    TORCH_CHECK(width > 0 && height > 0 && width < INT_MAX - Tile && height < INT_MAX - Tile,
                "invalid image dimensions");
    TORCH_CHECK(near_plane > 0 && far_plane > near_plane, "invalid clipping planes");
    TORCH_CHECK(image_ids.device() == geometry.device() && image_ids.is_contiguous()
                && image_ids.scalar_type() == torch::kInt64 && image_ids.dim() == 1
                && image_ids.size(0) == geometry.size(0), "invalid image IDs");
    TORCH_CHECK(radii.device() == geometry.device() && radii.is_contiguous()
                && radii.scalar_type() == torch::kInt32 && radii.dim() == 1
                && radii.size(0) == geometry.size(0), "invalid radii");
    c10::cuda::CUDAGuard guard(geometry.device());
    auto stream = c10::cuda::getCurrentCUDAStream(geometry.get_device()).stream();
    int count = geometry.size(0), images = backgrounds.size(0);
    int tw = (width + Tile - 1) / Tile, th = (height + Tile - 1) / Tile;
    TORCH_CHECK(int64_t(images) * tw * th <= INT_MAX, "image tile count exceeds int32");
    auto ints = geometry.options().dtype(torch::kInt32);
    auto longs = geometry.options().dtype(torch::kInt64);
    auto bytes = geometry.options().dtype(torch::kUInt8);
    auto maps = torch::zeros({images, height, width, MapSize}, geometry.options());
    auto state = torch::zeros({images, height, width, 3}, geometry.options());
    auto contributors = torch::zeros({images, height, width}, ints);
    auto ranges = torch::zeros({int64_t(images) * tw * th, 2}, ints);
    auto point_list = torch::empty({0}, ints);
    if (count > 0) {
        auto counts = torch::empty({count}, longs), offsets = torch::empty({count}, longs);
        countTiles<<<(count + 255) / 256, 256, 0, stream>>>(count, tw, th,
            geometry.data_ptr<float>(), radii.data_ptr<int>(), counts.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        size_t scan_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes,
            counts.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), count, stream));
        auto temp = torch::empty({int64_t(scan_bytes)}, bytes);
        C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(temp.data_ptr(), scan_bytes,
            counts.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), count, stream));
        int64_t intersections = 0;
        C10_CUDA_CHECK(cudaMemcpyAsync(&intersections, offsets.data_ptr<int64_t>() + count - 1,
            sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        TORCH_CHECK(intersections <= INT_MAX, "tile intersection count exceeds int32");
        if (intersections > 0) {
            auto keys = torch::empty({intersections}, longs), sorted_keys = torch::empty_like(keys);
            auto values = torch::empty({intersections}, ints);
            point_list = torch::empty_like(values);
            emitTiles<<<(count + 255) / 256, 256, 0, stream>>>(count, tw, th, geometry.data_ptr<float>(),
                radii.data_ptr<int>(), image_ids.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
                reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()), values.data_ptr<int>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            size_t sort_bytes = 0;
            C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes,
                reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()),
                reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
                values.data_ptr<int>(), point_list.data_ptr<int>(), int(intersections), 0, 64, stream));
            auto sort_temp = torch::empty({int64_t(sort_bytes)}, bytes);
            C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(sort_temp.data_ptr(), sort_bytes,
                reinterpret_cast<uint64_t*>(keys.data_ptr<int64_t>()),
                reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
                values.data_ptr<int>(), point_list.data_ptr<int>(), int(intersections), 0, 64, stream));
            findRanges<<<(intersections + 255) / 256, 256, 0, stream>>>(intersections,
                reinterpret_cast<uint64_t*>(sorted_keys.data_ptr<int64_t>()),
                reinterpret_cast<int2*>(ranges.data_ptr<int>()));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    #define FORWARD(NEURAL) renderForward<NEURAL><<<images * tw * th, dim3(Tile, Tile), 0, stream>>>( \
        width, height, tw, geometry.data_ptr<float>(), appearance.data_ptr<float>(), \
        opacities.data_ptr<float>(), primitive_ids.data_ptr<int64_t>(), backgrounds.data_ptr<float>(), \
        point_list.data_ptr<int>(), reinterpret_cast<int2*>(ranges.data_ptr<int>()), \
        near_plane, far_plane, maps.data_ptr<float>(), state.data_ptr<float>(), contributors.data_ptr<int>())
    if (neural) { FORWARD(true); } else { FORWARD(false); }
    #undef FORWARD
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {maps, point_list, ranges, state, contributors};
}

// Allocate input gradients and dispatch reverse compositing on the active CUDA stream.
std::vector<torch::Tensor> rasterizeBackward(
    torch::Tensor geometry, torch::Tensor appearance, torch::Tensor opacities,
    torch::Tensor primitive_ids, torch::Tensor backgrounds, torch::Tensor point_list,
    torch::Tensor ranges, torch::Tensor state, torch::Tensor contributors,
    torch::Tensor grad_maps, int64_t width, int64_t height, bool neural,
    double near_plane, double far_plane) {
    checkBuffers(geometry, appearance, opacities, primitive_ids, backgrounds, neural);
    checkFloat(grad_maps, geometry);
    c10::cuda::CUDAGuard guard(geometry.device());
    auto stream = c10::cuda::getCurrentCUDAStream(geometry.get_device()).stream();
    auto gg = torch::zeros_like(geometry), ga = torch::zeros_like(appearance);
    auto go = torch::zeros_like(opacities), gb = torch::zeros_like(backgrounds);
    int tw = (width + Tile - 1) / Tile, th = (height + Tile - 1) / Tile;
    #define BACKWARD(NEURAL) renderBackward<NEURAL><<<backgrounds.size(0) * tw * th, dim3(Tile, Tile), 0, stream>>>( \
        width, height, tw, geometry.data_ptr<float>(), appearance.data_ptr<float>(), \
        opacities.data_ptr<float>(), primitive_ids.data_ptr<int64_t>(), backgrounds.data_ptr<float>(), \
        point_list.data_ptr<int>(), reinterpret_cast<int2*>(ranges.data_ptr<int>()), \
        state.data_ptr<float>(), contributors.data_ptr<int>(), grad_maps.data_ptr<float>(), \
        near_plane, far_plane, gg.data_ptr<float>(), ga.data_ptr<float>(), go.data_ptr<float>(), gb.data_ptr<float>())
    if (neural) { BACKWARD(true); } else { BACKWARD(false); }
    #undef BACKWARD
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gg, ga, go, gb};
}
