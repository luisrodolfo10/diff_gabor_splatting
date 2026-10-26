# Rendering contracts

## Geometry and cameras

All rendering inputs are contiguous CUDA float32 tensors on one device. There
are no implicit device or dtype conversions. All values must be finite. Leading
scene dimensions must match exactly; no scene/camera broadcasting is implicit.
Each scene has N primitives; each scene has C cameras. N may be zero; C and every
scene dimension must be positive. Images within a call share width and height.

`means[..., N, 3]` are world-space positions. `quaternions[..., N, 4]` use wxyz,
are unit within 2e-4 numerical tolerance, and are normalized differentiably before
conversion to rotations. The first two rotation columns span the tangent plane;
its third column is the normal. `scales[..., N, 2]` are positive standard deviations
along those two local axes, in world units. `opacities[..., N]` are activated values
in [0,1]; apply your model's sigmoid before calling the renderer. Scales are not
logged inputs. No densification or parameter activation happens inside the package.

`view_matrices[..., C, 4, 4]` are rigid right-handed world-to-camera transforms,
used on column vectors, with last row [0,0,0,1]. Camera coordinates have +x to the
right, +y down, and +z forward. Rotations must be orthonormal with determinant +1.
`intrinsics[..., C, 3, 3]` have last row [0,0,1] and positive fx/fy. The upper-left
2x2 block can contain skew. Positions x=0..W-1, y=0..H-1 denote pixel centers;
there is no hidden half-pixel offset. Set principal points accordingly, for
example ((W-1)/2, (H-1)/2) for an exactly centered image.

The local scaled plane maps to pixels through
`T = K @ [R_camera @ tangent_u, R_camera @ tangent_v, camera_mean]`.
Ray/plane intersection gives local coordinates (u,v). A perspective conic bounds
the three-sigma local disk, with a minimum screen radius of 3*sqrt(1/2). The
Gaussian's exponent uses the minimum of u²+v² and twice the squared pixel distance
from the conic center. This is the upstream 2DGS low-pass filter.

Primitives with center depth <= near_plane or >= far_plane, degenerate conics,
exactly edge-on facing tests, or empty tile bounds are culled. Defaults are
near_plane=0.2 and far_plane=100, in world/camera length units. Classic depth uses
the surface intersection for the surface branch of the low-pass filter and the
center depth otherwise; neural depth always uses the intersection, matching the
respective upstream methods. Intersections closer than near_plane are skipped.

## Appearance

Classic Gabor takes two RGB colors and four weights, magnitudes, and phases per
primitive. Candidate f=0,1,2,3 has local direction
`d_f=(cos(pi*f/4), sin(pi*f/4))` and wave
`cos(2*pi*magnitude_f*dot(d_f,(u,v))+phase_f)`. Magnitudes are nonnegative cycles
per scaled local unit, phases are radians. Weights are nonnegative and sum to one
within 2e-4 tolerance. The first color has mixture coefficient
`sum(weight_f*(1+wave_f)/2)`; the second uses `(1-wave_f)/2`. RGB values are used
as supplied, with no hidden clamp.

With `sh_degree=0..3`, both colors instead have shape
`[..., N, (degree+1)**2, 3]`. Real SH use the Inria ordering and constants. Direction
is the normalized world vector from camera center to primitive center. SH RGB is
`clamp_min(evaluate_sh(coefficients, direction)+0.5, 0)`. Color coefficients are
shared across cameras; evaluated colors are camera-specific.

Neural Gabor uses one 5→6→3 network per primitive. Input weights are `[6,5]`,
input biases `[6]`, output weights `[3,6]`, and output biases `[3]` after the scene
and primitive dimensions. The input is `(u/3,v/3,direction_x,direction_y,direction_z)`;
the direction is the same world direction as above. Hidden activation is
`sin(30*(W1@input+b1))`; output is `sigmoid(W2@hidden+b2)`. Matrix rows correspond
to output neurons. Networks are evaluated for each contributing pixel, rather
than predicting a single color at the primitive center.

For SIREN initialization, the first weight layer can use uniform [-1/5,1/5];
the output layer can use uniform [-sqrt(6/6)/30,sqrt(6/6)/30]. Bias tensors are
explicit inputs. The package imposes no hidden initialization or optimization.

## Outputs and compositing

`rgb` has shape `[..., C, H, W, 3]`; `alpha` has shape `[..., C, H, W, 1]`.
Primitives are sorted by center camera depth within each image tile. Equal-depth
keys preserve original primitive order through CUB's stable sort. Alpha per
contribution is `min(0.99, opacity*exp(-rho/2))`. Alpha below 1/255 is skipped.
Rendering stops before a contribution that would reduce transmittance below
1e-4, following the upstream convention. RGB is front-to-back weighted color plus
remaining transmittance times background. Background defaults to zero and otherwise
has shape `[..., C, 3]`.

Named metadata:

- `accumulated_depth`: `[..., C, H, W, 1]`, sum(weight*intersection_depth), not
  normalized by alpha. Expected depth can be computed by dividing by alpha on
  nonempty pixels; empty accumulated depth is zero.
- `normals`: `[..., C, H, W, 3]`, alpha-weighted camera-space facing normals.
  Normals face toward the camera and are not renormalized after accumulation.
- `distortion`: `[..., C, H, W, 1]`, sum over i>j of
  `weight_i*weight_j*(m_i-m_j)**2`, with
  `m=far/(far-near)*(1-near/depth)`. This is the upstream depth-distortion convention
  without an additional factor of two. Float32 cancellation can produce small
  negative roundoff near zero; the renderer does not silently clamp it.
- `radii`: int32 screen radii, `[..., C, N]` in dense mode or `[Q]` in packed mode.
  Zero indicates a culled pair.
- `projected_means`: `[Q, 2]`, conic centers in pixels. Dense mode has
  Q=product(scene dimensions)*C*N, including culled pairs; packed mode has only
  visible pairs. This tensor is used in the compositor graph. Call `retain_grad()`
  on it before backward to inspect its screen-filter contribution. This gradient
  does not include the separate ray/plane-transform path of a surface primitive.
- `scene_ids`, `camera_ids`, `primitive_ids`: int64 `[Q]` indices aligned with
  projected means. Scene IDs flatten all leading scene dimensions in row-major order.
- `packed`: whether invisible pairs were compacted before tile binning.

Dense and packed modes return identically shaped images. Packing describes
visible projected pairs, not variable-length scene inputs. Projection is dense
before compaction in this first implementation; packing reduces compositor and
binning storage, not the initial projection/autograd graph.

## Differentiability and execution

RGB, alpha, accumulated depth, normals, and distortion support first-order
reverse-mode gradients to geometry, opacity, appearance, and supplied backgrounds.
Camera and intrinsic gradients flow through projection and appearance direction;
optimize camera poses through a rigid pose parameterization and intrinsics through
a pinhole parameterization. Discrete culling, sorting, tile membership, radius,
normal orientation, and early termination have no derivative. Clamp derivatives
are zero in the saturated alpha branch. Behavior exactly at discrete/clamp
boundaries is not promised. Higher-order derivatives and forward AD are unsupported.

Fused scene/camera batching is inspired by [gsplat](https://docs.gsplat.studio/main/);
its paper is cited in [references.bib](../references.bib).
All native work uses the current device and current CUDA stream. One CUB scan,
one image-aware tile/depth sort, and one compositor launch serve every scene and
camera in a call. Native rendering uses float32 atomics in backward, so sums may
vary slightly between runs and between fused/loop implementations. Numerical
input checks synchronize once per call. Allocating variable-size tile intersection
storage also requires a stream synchronization after the scan. CUDA graph capture
is not supported. There is no CPU or alternative-backend fallback.

Near-grazing ray/plane intersections are ill-conditioned: small float32 changes
can cause large local-coordinate and neural color derivatives. This implementation
does not clamp local coordinates or silently replace the surface model. Expect
larger relative gradient variation in those cases. The compact performance
benchmark uses moderately tilted surfaces to measure a reproducible workload.
