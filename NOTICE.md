# Attribution and source provenance

This package implements surface projection, low-pass filtering and Gabor appearance
following the following rasterizers by Haato Watanabe and collaborators, derived
from 2D Gaussian Splatting and the Inria Gaussian Splatting codebase:

- https://github.com/haato-w/diff-gabor-rasterization
  commit bf33b99c5a76e0d1b83f97914caf250422a90509
- https://github.com/haato-w/diff-neural-gabor-rasterization
  commit 44f4501bcc08c021dd68592ade593f68bb36e284

The fused scene/camera batching and image-aware tile binning design is inspired
by gsplat: https://github.com/nerfstudio-project/gsplat.
See references.bib for the gsplat paper citation. Package implementation by
Luis Rodolfo Macias includes the shared surface/Gabor compositor and its
reverse-mode derivatives.

Upstream notices:

Copyright (C) 2023, Inria
GRAPHDECO research group, https://team.inria.fr/graphdeco
All rights reserved.

This software is free for non-commercial, research and evaluation use under the
terms of LICENSE.md. For inquiries contact george.drettakis@inria.fr.

Retain this notice and LICENSE.md when redistributing this package.
