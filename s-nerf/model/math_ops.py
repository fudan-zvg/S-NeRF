import torch
import math
import numpy as np


def safe_trig_helper(x, fn, t=100 * math.pi):
  return fn(torch.where(torch.abs(x) < t, x, x % t))


def safe_sin(x):
  """jnp.sin() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, torch.sin)


def safe_cos(x):
  """jnp.cos() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, torch.cos)

def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
  """Piecewise-Constant PDF sampling from sorted bins.
  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
    weights: jnp.ndarray(float32), [batch_size, num_bins].
    num_samples: int, the number of samples.
    randomized: bool, use randomized samples.
  Returns:
    t_samples: jnp.ndarray(float32), [batch_size, num_samples].
  """
  # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
  # avoids NaNs when the input is zeros or small, but has no effect otherwise.
  eps = 1e-5
  weight_sum = torch.sum(weights, dim=-1, keepdim=True)
  padding = torch.maximum(torch.zeros_like(eps - weight_sum), eps - weight_sum)
  weights = weights + padding / weights.shape[-1]
  weight_sum = weight_sum + padding

  # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
  # starts with exactly 0 and ends with exactly 1.
  pdf = weights / weight_sum
  cdf = torch.minimum(torch.ones_like(torch.cumsum(pdf[..., :-1], dim=-1)), torch.cumsum(pdf[..., :-1], dim=-1))
  cdf = torch.cat([
      torch.zeros(list(cdf.shape[:-1]) + [1], device=cdf.device), cdf,
      torch.ones(list(cdf.shape[:-1]) + [1], device=cdf.device)
  ],
                        dim=-1)

  # Draw uniform samples.
  if randomized:
    s = 1 / num_samples
    u = torch.arange(num_samples, device=cdf.device) * s
    u = u + torch.empty(size=list(cdf.shape[:-1]) + [num_samples], device=cdf.device).uniform_(to=s - torch.finfo(torch.float32).eps)
    # `u` is in [0, 1) --- it can be zero, but it can never be 1.
    u = torch.minimum(u, torch.ones_like(u) - torch.finfo(torch.float32).eps)
  else:
    # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
    u = torch.linspace(0., 1. - torch.finfo(torch.float32).eps, num_samples,device=cdf.device)
    u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

  # Identify the location in `cdf` that corresponds to a random sample.
  # The final `True` index in `mask` will be the start of the sampled interval.
  mask = u[..., None, :] >= cdf[..., :, None]

  def find_interval(x):
    # Grab the value where `mask` switches from True to False, and vice versa.
    # This approach takes advantage of the fact that `x` is sorted.
    x0, _ = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
    x1, _ = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
    return x0, x1

  bins_g0, bins_g1 = find_interval(bins)
  cdf_g0, cdf_g1 = find_interval(cdf)

  t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
  samples = bins_g0 + t * (bins_g1 - bins_g0)
  return samples

def mse_to_psnr(mse):
  """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
  return -10. / math.log(10.) * torch.log(mse)