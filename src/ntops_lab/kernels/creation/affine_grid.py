import functools

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


@functools.cache
def _component_kernel(height, width):
    scale_y = 2.0 / float(height)
    scale_x = 2.0 / float(width)

    def arrangement(theta_row, out_component, scale_y_tensor, scale_x_tensor):
        theta_arr = theta_row[:, None, None, :].expand((-1, height, width, -1)).flatten(end_dim=3)
        out_arr = out_component.flatten()
        return theta_arr.tile((1, 3)), out_arr.tile((1,)), scale_y_tensor, scale_x_tensor

    def application(theta, out, scale_y_tensor, scale_x_tensor):
        x = ((out.offsets(2) + 0.5) * scale_x_tensor) - 1.0
        y = ((out.offsets(1) + 0.5) * scale_y_tensor) - 1.0
        t0 = ntl.sum(theta * (theta.offsets(1) == 0), axis=1)
        t1 = ntl.sum(theta * (theta.offsets(1) == 1), axis=1)
        t2 = ntl.sum(theta * (theta.offsets(1) == 2), axis=1)
        out = t0 * x + t1 * y + t2

    return ninetoothed.make(
        arrangement,
        application,
        (
            Tensor(2),
            Tensor(3),
            Tensor(0, constexpr=True, value=scale_y, name="scale_y"),
            Tensor(0, constexpr=True, value=scale_x, name="scale_x"),
        ),
        kernel_name=f"ntops_lab_affine_grid_component_{height}x{width}",
        max_num_configs=1,
    )


def run(theta, size, align_corners=False):
    if align_corners or len(size) != 4:
        raise ValueError("affine_grid currently supports 2D align_corners=False")
    height, width = int(size[-2]), int(size[-1])
    out = torch.empty((theta.shape[0], height, width, 2), device=theta.device, dtype=theta.dtype)
    kernel = _component_kernel(height, width)
    scale_y = 2.0 / float(height)
    scale_x = 2.0 / float(width)
    kernel(theta[:, 0, :], out[..., 0], scale_y, scale_x)
    kernel(theta[:, 1, :], out[..., 1], scale_y, scale_x)
    return out
