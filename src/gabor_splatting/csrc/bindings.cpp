// Copyright (C) 2023, Inria. See LICENSE.md and NOTICE.md.
#include <torch/extension.h>
#include "rasterize.h"

// Register the private native rendering and reverse-mode entry points.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("rasterize_forward", &rasterizeForward);
    module.def("rasterize_backward", &rasterizeBackward);
    module.def("evaluate_sh_forward", &evaluateSHForward);
    module.def("evaluate_sh_backward", &evaluateSHBackward);
}
