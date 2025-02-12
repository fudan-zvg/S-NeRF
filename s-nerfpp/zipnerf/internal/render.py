from internal import stepfun
import torch
import torch.nn.functional as F


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = d[..., None, :] * t_mean[..., None]
    eps = torch.finfo(d.dtype).eps
    # eps = 1e-3
    d_mag_sq = torch.sum(d ** 2, dim=-1, keepdim=True).clamp_min(eps)

    if diag:
        d_outer_diag = d ** 2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1], device=d.device)
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and base_radius is the
  radius at dist=1. Doesn't assume `d` is normalized.

  Args:
    d: the axis of the cone
    t0: the starting distance of the frustum.
    t1: the ending distance of the frustum.
    base_radius: the scale of the radius as a function of distance.
    diag: whether or the Gaussian will be diagonal or full-covariance.
    stable: whether or not to use the stable computation described in
      the paper (setting this to False will cause catastrophic failure).

  Returns:
    a Gaussian (mean and covariance).
  """
    if stable:
        # Equation 7 in the paper (https://arxiv.org/abs/2103.13415).
        mu = (t0 + t1) / 2  # The average of the two `t` values.
        hw = (t1 - t0) / 2  # The half-width of the two `t` values.
        eps = torch.finfo(d.dtype).eps
        # eps = 1e-3
        t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2).clamp_min(eps)
        denom = (3 * mu ** 2 + hw ** 2).clamp_min(eps)
        t_var = (hw ** 2) / 3 - (4 / 15) * hw ** 4 * (12 * mu ** 2 - hw ** 2) / denom ** 2
        r_var = (mu ** 2) / 4 + (5 / 12) * hw ** 2 - (4 / 15) * (hw ** 4) / denom
    else:
        # Equations 37-39 in the paper.
        t_mean = (3 * (t1 ** 4 - t0 ** 4)) / (4 * (t1 ** 3 - t0 ** 3))
        r_var = 3 / 20 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_mosq = 3 / 5 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_var = t_mosq - t_mean ** 2
    r_var *= base_radius ** 2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
    """Approximate a cylinder as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and radius is the
  radius. Does not renormalize `d`.

  Args:
    d: the axis of the cylinder
    t0: the starting distance of the cylinder.
    t1: the ending distance of the cylinder.
    radius: the radius of the cylinder
    diag: whether or the Gaussian will be diagonal or full-covariance.

  Returns:
    a Gaussian (mean and covariance).
  """
    t_mean = (t0 + t1) / 2
    r_var = radius ** 2 / 4
    t_var = (t1 - t0) ** 2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)

# ori guchun
# def cast_rays(tdist, origins, viewdirs, radii, rand=True, n=7, m=3, std_scale=0.35, batch=None):
#     """Cast rays (cone- or cylinder-shaped) and featurize sections of it.
#
#   Args:
#     tdist: float array, the "fencepost" distances along the ray.
#     origins: float array, the ray origin coordinates.
#     directions: float array, the ray direction vectors.
#     radii: float array, the radii (base radii for cones) of the rays.
#     ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
#     diag: boolean, whether or not the covariance matrices should be diagonal.
#
#   Returns:
#     a tuple of arrays of means and covariances.
#   """
#     t0 = tdist[..., :-1]
#     t1 = tdist[..., 1:]
#
#     j = torch.arange(n, device=tdist.device)
#     t = t0[..., None] + (t1[..., None] - t0[..., None]) * (j + 0.5) / n
#     deg = torch.broadcast_to(2 * torch.pi * m * j / n, t.shape)
#     if rand:
#         deg = deg + torch.rand_like(deg) * torch.pi * 2
#     means = torch.stack([
#         radii[..., None] * t * torch.cos(deg) / 2,
#         radii[..., None] * t * torch.sin(deg) / 2,
#         t
#     ], dim=-1)
#     stds = std_scale * radii[..., None] * t
#
#     rand_vec = torch.randn_like(viewdirs)
#     ortho1 = F.normalize(torch.cross(viewdirs, rand_vec, dim=-1), dim=-1)
#     ortho2 = F.normalize(torch.cross(viewdirs, ortho1, dim=-1), dim=-1)
#     basis_matrix = torch.stack([ortho1, ortho2, viewdirs], dim=-1)
#     means = torch.matmul(means, basis_matrix[..., None, :, :].transpose(-1, -2))
#     means = means + origins[..., None, None, :]
#     # import trimesh
#     # trimesh.Trimesh(means.reshape(-1, 3).detach().cpu().numpy()).export("test.ply", "ply")
#
#     return means, stds
def cast_rays(tdist, origins, directions, radii, rand=True, n=7, m=3, std_scale=0.35, batch=None):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

  Args:
    tdist: float array, the "fencepost" distances along the ray.
    origins: float array, the ray origin coordinates.
    directions: float array, the ray direction vectors.
    radii: float array, the radii (base radii for cones) of the rays.
    ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
    diag: boolean, whether or not the covariance matrices should be diagonal.

  Returns:
    a tuple of arrays of means and covariances.
  """
    t0 = tdist[..., :-1]
    t1 = tdist[..., 1:]

    j = torch.arange(n, device=tdist.device)
    t = t0[..., None] + (t1[..., None] - t0[..., None]) * (j + 0.5) / n
    deg = torch.broadcast_to(2 * torch.pi * m * j / n, t.shape)
    if rand:
        deg = deg + torch.rand_like(deg) * torch.pi * 2
    means = torch.stack([
        radii[..., None] * t * torch.cos(deg) / 2,
        radii[..., None] * t * torch.sin(deg) / 2,
        t
    ], dim=-1)
    stds = std_scale * radii[..., None] * t
    base_x, base_y = batch['base_x'], batch['base_y']

    # rand_vec = torch.randn_like(directions)
    # ortho1 = F.normalize(torch.cross(viewdirs, rand_vec, dim=-1), dim=-1)
    # ortho2 = F.normalize(torch.cross(viewdirs, ortho1, dim=-1), dim=-1)
    basis_matrix = torch.stack([base_x, base_y, directions], dim=-1)
    means = torch.matmul(means, basis_matrix[..., None, :, :].transpose(-1, -2))
    means = means + origins[..., None, None, :]
    # import trimesh
    # trimesh.Trimesh(means.reshape(-1, 3).detach().cpu().numpy()).export("test.ply", "ply")

    return means, stds

def compute_alpha_weights(density, tdist, dirs, opaque_background=False):
    """Helper function for computing alpha compositing weights."""
    t_delta = tdist[..., 1:] - tdist[..., :-1]
    delta = t_delta * torch.norm(dirs[..., None, :], dim=-1)
    density_delta = density * delta

    if opaque_background:
        # Equivalent to making the final t-interval infinitely wide.
        density_delta = torch.cat([
            density_delta[..., :-1],
            torch.full_like(density_delta[..., -1:], torch.inf)
        ], dim=-1)

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    weights = alpha * trans
    return weights, alpha, trans


def volumetric_rendering(rgbs,
                         weights,
                         tdist,
                         bg_rgbs,
                         t_far,
                         compute_extras,
                         semantic=None,
                         extras=None):
    """Volumetric Rendering Function.

  Args:
    rgbs: color, [batch_size, num_samples, 3]
    weights: weights, [batch_size, num_samples].
    tdist: [batch_size, num_samples].
    bg_rgbs: the color(s) to use for the background.
    t_far: [batch_size, 1], the distance of the far plane.
    compute_extras: bool, if True, compute extra quantities besides color.
    extras: dict, a set of values along rays to render by alpha compositing.

  Returns:
    rendering: a dict containing an rgb image of size [batch_size, 3], and other
      visualizations if compute_extras=True.
  """
    eps = torch.finfo(rgbs.dtype).eps
    # eps = 1e-3
    rendering = {}

    acc = weights.sum(dim=-1)
    bg_w = (1 - acc[..., None]).clamp_min(0.)  # The weight of the background.
    rgb = (weights[..., None] * rgbs).sum(dim=-2) + bg_w * bg_rgbs
    rendering['rgb'] = rgb

    t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
    expectation = lambda x: (weights * x).sum(dim=-1) / acc.clamp_min(eps)
    # For numerical stability this expectation is computing using log-distance.
    rendering['depth'] = (
        torch.clip(
            torch.nan_to_num(torch.exp(expectation(torch.log(t_mids))), torch.inf),
            tdist[..., 0], tdist[..., -1]))
    # depth = (weights * t_mids).sum(dim=-1)
    # rendering['depth'] = depth

    if semantic is not None:
        ### no influences to density
        weights_sem = weights.clone().detach()
        sem = (weights_sem[..., None] * semantic).sum(dim=-2)
        rendering['semantic'] = sem

    if compute_extras:
        rendering['acc'] = acc

        if extras is not None:
            for k, v in extras.items():
                if v is not None:
                    rendering[k] = (weights[..., None] * v).sum(dim=-2)

        expectation = lambda x: (weights * x).sum(dim=-1) / acc.clamp_min(eps)
        t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
        # For numerical stability this expectation is computing using log-distance.
        rendering['distance_mean'] = (
            torch.clip(
                torch.nan_to_num(torch.exp(expectation(torch.log(t_mids))), torch.inf),
                tdist[..., 0], tdist[..., -1]))

        # Add an extra fencepost with the far distance at the end of each ray, with
        # whatever weight is needed to make the new weight vector sum to exactly 1
        # (`weights` is only guaranteed to sum to <= 1, not == 1).
        t_aug = torch.cat([tdist, t_far], dim=-1)
        weights_aug = torch.cat([weights, bg_w], dim=-1)

        ps = [5, 50, 95]
        distance_percentiles = stepfun.weighted_percentile(t_aug, weights_aug, ps)

        for i, p in enumerate(ps):
            s = 'median' if p == 50 else 'percentile_' + str(p)
            rendering['distance_' + s] = distance_percentiles[..., i]

    return rendering
