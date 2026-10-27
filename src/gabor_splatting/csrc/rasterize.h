// Copyright (C) 2023, Inria. See LICENSE.md and NOTICE.md.
#pragma once
#include <torch/types.h>
#include <vector>

std::vector<torch::Tensor> rasterizeForward(
    torch::Tensor geometry, torch::Tensor appearance, torch::Tensor opacities,
    torch::Tensor primitive_ids, torch::Tensor image_ids, torch::Tensor radii,
    torch::Tensor backgrounds, int64_t width, int64_t height, bool neural,
    double near_plane, double far_plane);
std::vector<torch::Tensor> rasterizeBackward(
    torch::Tensor geometry, torch::Tensor appearance, torch::Tensor opacities,
    torch::Tensor primitive_ids, torch::Tensor backgrounds, torch::Tensor point_list,
    torch::Tensor ranges, torch::Tensor state, torch::Tensor contributors,
    torch::Tensor grad_maps, int64_t width, int64_t height, bool neural,
    double near_plane, double far_plane);

// Evaluate both classic SH colors and their first-order derivatives.
torch::Tensor evaluateSHForward(torch::Tensor first, torch::Tensor second,
                                torch::Tensor directions, int64_t degree);
std::vector<torch::Tensor> evaluateSHBackward(torch::Tensor first, torch::Tensor second,
    torch::Tensor directions, torch::Tensor grad_colors, int64_t degree);
