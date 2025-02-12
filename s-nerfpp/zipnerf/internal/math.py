import numpy as np
import torch


@torch.jit.script
def erf(x):
    return torch.sign(x) * torch.sqrt(1 - torch.exp(-4 / torch.pi * x ** 2))


def matmul(a, b):
    return (a[..., None] * b[..., None, :, :]).sum(dim=-2)
    # B,3,4,1  B,1,4,3

    # cause nan when fp16
    # return torch.matmul(a, b)


def safe_trig_helper(x, fn, t=100 * torch.pi):
    """Helper function used by safe_cos/safe_sin: mods x before sin()/cos()."""
    return fn(torch.where(torch.abs(x) < t, x, x % t))


def safe_cos(x):
    return safe_trig_helper(x, torch.cos)


def safe_sin(x):
    return safe_trig_helper(x, torch.sin)


def safe_exp(x):
    return torch.exp(x.clamp_max(88.))


def safe_exp_jvp(primals, tangents):
    """Override safe_exp()'s gradient so that it's large when inputs are large."""
    x, = primals
    x_dot, = tangents
    exp_x = safe_exp(x)
    exp_x_dot = exp_x * x_dot
    return exp_x, exp_x_dot


def log_lerp(t, v0, v1):
    """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
    if v0 <= 0 or v1 <= 0:
        raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
    lv0 = np.log(v0)
    lv1 = np.log(v1)
    return np.exp(np.clip(t, 0, 1) * (lv1 - lv0) + lv0)


def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
    """Continuous learning rate decay function.

  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.

  Args:
    step: int, the current optimization step.
    lr_init: float, the initial learning rate.
    lr_final: float, the final learning rate.
    max_steps: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.

  Returns:
    lr: the learning for current step 'step'.
  """
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
    else:
        delay_rate = 1.
    return delay_rate * log_lerp(step / max_steps, lr_init, lr_final)


def sorted_interp(x, xp, fp):
    """A TPU-friendly version of interp(), where xp and fp must be sorted."""

    # Identify the location in `xp` that corresponds to each `x`.
    # The final `True` index in `mask` is the start of the matching interval.
    mask = x[..., None, :] >= xp[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2).values
        x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2).values
        return x0, x1

    fp0, fp1 = find_interval(fp)
    xp0, xp1 = find_interval(xp)

    offset = torch.clip(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    ret = fp0 + offset * (fp1 - fp0)
    return ret


# def sorted_interp_quad(x, xp, fpdf, fcdf):
#     """interp in quadratic"""
#
#     # Identify the location in `xp` that corresponds to each `x`.
#     # The final `True` index in `mask` is the start of the matching interval.
#     mask = x[..., None, :] >= xp[..., :, None]
#
#     def find_interval(x):
#         # Grab the value where `mask` switches from True to False, and vice versa.
#         # This approach takes advantage of the fact that `x` is sorted.
#         x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2).values
#         x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2).values
#         return x0, x1
#
#     fpdf0, fpdf1 = find_interval(fpdf)
#     fcdf0, fcdf1 = find_interval(fcdf)
#     xp0, xp1 = find_interval(xp)
#
#     offset = torch.clip(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
#     ret = fcdf0 + (x - xp0) * (fpdf0 + fpdf1 * offset + fpdf0 * (1 - offset)) / 2
#     return ret


def sorted_interp_quad(x, xp, fpdf, fcdf):
    """interp in quadratic"""

    # Identify the location in `xp` that corresponds to each `x`.
    # The final `True` index in `mask` is the start of the matching interval.
    # mask = x[..., None, :] >= xp[..., :, None]
    # x_c = torch.cat([-torch.inf*torch.ones_like(xp[...,:1]),xp,torch.inf*torch.ones_like(xp[...,:1])], -1)
    # import pdb; pdb.set_trace()
    whole_x = torch.cat([xp, x], -1)
    _, sorted_idx = torch.sort(whole_x, -1)
    len = xp.shape[-1]
    valid_x = sorted_idx<len
    idx = torch.cumsum(valid_x, -1)
    valid_mask = sorted_idx>=len
    idx = idx[valid_mask].reshape(*x.shape).long()
    idx_0, idx_1 = (idx-1).clamp_min(0), idx.clamp_max(len-1)

    fpdf0, fpdf1 = torch.gather(fpdf, -1, idx_0), torch.gather(fpdf, -1, idx_1)
    fcdf0 = torch.gather(fcdf, -1, idx_0)
    xp0, xp1 = torch.gather(xp, -1, idx_0), torch.gather(xp, -1, idx_1)

    offset = torch.clip(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    ret = fcdf0 + (x - xp0) * (fpdf0 + fpdf1 * offset + fpdf0 * (1 - offset)) / 2
    return ret



    # def find_indices(x):
    #     # Grab the value where `mask` switches from True to False, and vice versa.
    #     # This approach takes advantage of the fact that `x` is sorted.
    #     x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2).values
    #     x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2).values
    #     return x0, x1
    #
    # fpdf0, fpdf1 = find_interval(fpdf)
    # fcdf0, fcdf1 = find_interval(fcdf)
    # xp0, xp1 = find_interval(xp)
    #
    # offset = torch.clip(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    # ret = fcdf0 + (x - xp0) * (fpdf0 + fpdf1 * offset + fpdf0 * (1 - offset)) / 2
    # return ret