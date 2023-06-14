import torch
import model.math_ops as math_ops
import math
import random
from functools import partial
# from torch.autograd.functional import jacobian
def Transform(x, near, far): return 1./((1-x)/near+x/far)
def Transform_log(x, near, far): return near*torch.exp(x*torch.log(far/near))
def Transform_linear(x, near, far): return near*(1-x)+far*x


def pos_enc(x, min_deg, max_deg, append_identity=True):
    scales = torch.tensor(
        [2**i for i in range(min_deg, max_deg)], device=x.device)
    xb = torch.reshape((x[..., None, :] * scales[:, None]),
                       list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)
    else:
        return four_feat


def expected_sin(x, x_var):
    y = torch.exp(-0.5 * x_var) * math_ops.safe_sin(x)
    y_var = torch.maximum(
        torch.zeros_like(x_var, device=x_var.device), 0.5 * (1 - torch.exp(-2 * x_var) * math_ops.safe_cos(2 * x)) - y ** 2)
    return y, y_var


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    mean = d[..., None, :] * t_mean[..., None]

    small_tensor = torch.ones_like(
        torch.sum(d**2, dim=-1, keepdims=True)) * 1e-10
    d_mag_sq = torch.maximum(
        small_tensor, torch.sum(d**2, dim=-1, keepdims=True))

    if diag:
        d_outer_diag = d**2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1])
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
        t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                          (3 * mu**2 + hw**2)**2)
        r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                                  (hw**4) / (3 * mu**2 + hw**2))
    else:
        t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
        r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
        t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
        t_var = t_mosq - t_mean**2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
    t_mean = (t0 + t1) / 2
    r_var = radius**2 / 4
    t_var = (t1 - t0)**2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(t_vals, origins, directions, radii, ray_shape, diag=True):
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    if ray_shape == 'cone':
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == 'cylinder':
        gaussian_fn = cylinder_to_gaussian
    else:
        assert False
    means, covs = gaussian_fn(directions, t0, t1, radii, diag)
    means = means + origins[..., None, :]
    return means, covs


def integrated_pos_enc(x_coord, min_deg, max_deg, diag=True, device='0'):
    if diag:
        x, x_cov_diag = x_coord
        scales = torch.tensor(
            [2**i for i in range(min_deg, max_deg)], device=device)
        shape = list(x.shape[:-1]) + [-1]
        y = torch.reshape(x[..., None, :] * scales[:, None], shape)
        # import pdb;pdb.set_trace()
        y_var = torch.reshape(
            x_cov_diag[..., None, :] * scales[:, None]**2, shape)
    else:
        x, x_cov = x_coord
        # import pdb;pdb.set_trace()
        num_dims = x.shape[-1]
        basis = torch.cat(
            [2**i * torch.eye(num_dims) for i in range(min_deg, max_deg)], 1).to(x.device).float()
        y = torch.matmul(x.float(), basis)

        y_var = torch.sum((x_cov @ basis) * basis, -2)
        # import pdb;pdb.set_trace()
    # print('y',y)
    # print('y_var',y_var)
    return expected_sin(
        torch.cat([y, y + 0.5 * math.pi], dim=-1),
        torch.cat([y_var] * 2, dim=-1))[0]


def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):
    # this t_vals distributes in [0,1]
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]

    delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ],
        dim=-1))
    weights = alpha * trans

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    acc = weights.sum(dim=-1)
    # TODO:calculate the right distance in the case of warp_sampling
    # distance = (weights * t_mids).sum(dim=-1)/acc
    distance = (weights * t_mids).sum(dim=-1)
    # import pdb;pdb.set_trace()
    distance = torch.clip(
        torch.nan_to_num(distance, float('inf')), t_vals[:, 0], t_vals[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - acc[..., None])
    return comp_rgb, distance, acc, weights


def real_volumetric_rendering(rgb, density, t_vals, dirs, raw_semantic, white_bkgd, near, far, transform_idx=0):
    # transform the t_vals from [0,1] to near and far, to get true depth.
    T = transform(transform_idx)
    t_vals = T(t_vals, near, far)
    # import pdb;pdb.set_trace()
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]

    delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ],
        dim=-1))
    weights = alpha * trans

    if rgb is not None:
        comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    else:
        comp_rgb = None
    if raw_semantic is not None:
        semantic = (weights[..., None] * raw_semantic).sum(dim=-2)
    else:
        semantic = None

    acc = weights.sum(dim=-1)
    # TODO:calculate the right distance in the case of warp_sampling
    # do not apply normalization on the distance
    distance = (weights * t_mids).sum(dim=-1)
    # import pdb;pdb.set_trace()
    distance = torch.clip(
        torch.nan_to_num(distance, float('inf')), t_vals[:, 0], t_vals[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - acc[..., None])
    return comp_rgb, distance, acc, weights, semantic


def sample_along_rays(origins, directions, radii, num_samples, near, far,
                      randomized, lindisp, ray_shape):
    batch_size = origins.shape[0]
    device = origins.device
    t_vals = torch.linspace(0., 1., num_samples + 1, device=device)
    if lindisp:
        t_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    else:
        t_vals = near * (1. - t_vals) + far * t_vals
    # import pdb;pdb.set_trace()
    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples + 1, device=device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        t_vals = torch.broadcast_to(t_vals, [batch_size, num_samples + 1])
    means, covs = cast_rays(t_vals, origins, directions, radii, ray_shape)
    return t_vals, (means, covs)


def resample_along_rays(origins, directions, radii, t_vals, weights,
                        randomized, ray_shape, stop_grad, resample_padding):
    weights_pad = torch.cat([
        weights[..., :1],
        weights,
        weights[..., -1:],
    ],
        dim=-1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    # Add in a constant (the sampling function will renormalize the PDF).
    weights = weights_blur + resample_padding

    new_t_vals = math_ops.sorted_piecewise_constant_pdf(
        t_vals,
        weights,
        t_vals.shape[-1],
        randomized,
    )
    if stop_grad:
        new_t_vals = new_t_vals.detach()
    means, covs = cast_rays(new_t_vals, origins, directions, radii, ray_shape)
    return new_t_vals, (means, covs)


def real_resample_along_rays(origins, directions, radii, t_vals, weights,
                             randomized, ray_shape, stop_grad, resample_padding, near, far, transform_idx):
    weights_pad = torch.cat([
        weights[..., :1],
        weights,
        weights[..., -1:],
    ],
        dim=-1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    # Add in a constant (the sampling function will renormalize the PDF).
    weights = weights_blur + resample_padding
    T = transform(transform_idx=transform_idx)
    t_vals = T(t_vals, near, far)
    new_t_vals = math_ops.sorted_piecewise_constant_pdf(
        t_vals,
        weights,
        t_vals.shape[-1],
        randomized,
    )
    if stop_grad:
        new_t_vals = new_t_vals.detach()
    means, covs = cast_rays(new_t_vals, origins, directions, radii, ray_shape)
    return new_t_vals, (means, covs)


def warp_sample_along_rays(origins, directions, radii, num_samples, near, far,
                           randomized, lindisp, ray_shape, viewc, fn_idx=1, radius=3., transform_idx=0):
    # try g(x) = 1/x ;t=g^{-1}(s*g(far)+(1-s)*g(near))
    # f(x)=W(x-viewc)*(x-viewc)
    N = origins.shape[0]

    batch_size = origins.shape[0]  # N
    device = origins.device
    # s is the sampling space
    s_vals = torch.linspace(0., 1., num_samples + 1, device=device)

    if randomized:
        mids = 0.5 * (s_vals[..., 1:] + s_vals[..., :-1])
        upper = torch.cat([mids, s_vals[..., -1:]], -1)
        lower = torch.cat([s_vals[..., :1], mids], -1)
        s_rand = torch.rand(batch_size, num_samples + 1, device=device)
        s_vals = lower + (upper - lower) * s_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        s_vals = torch.broadcast_to(s_vals, [batch_size, num_samples + 1])

    f_means, f_covs = sample2enc(s_vals, origins, directions, radii, ray_shape, near,
                                 far, num_samples, fn_idx, viewc=viewc, radius=radius, transform_idx=transform_idx)
    return s_vals, (f_means, f_covs)


def warp_resample_along_rays(origins, directions, radii, t_vals, weights,
                             randomized, N_fine, ray_shape, stop_grad, resample_padding, viewc, near, far, fn_idx=1, radius=3., transform_idx=0.):
    weights_pad = torch.cat([
        weights[..., :1],
        weights,
        weights[..., -1:],
    ],
        dim=-1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    # Add in a constant (the sampling function will renormalize the PDF).
    weights = weights_blur + resample_padding

    new_t_vals = math_ops.sorted_piecewise_constant_pdf(
        t_vals,
        weights,
        N_fine,
        randomized,
    )
    if stop_grad:
        new_t_vals = new_t_vals.detach()
    num_samples = new_t_vals.shape[1]-1

    f_means, f_covs = sample2enc(new_t_vals, origins, directions, radii, ray_shape, near,
                                 far, num_samples, fn_idx, viewc=viewc, radius=radius, transform_idx=transform_idx)
    return new_t_vals, (f_means, f_covs)


def Jacobi_f(x, far):
    ln = torch.linalg.norm(x, axis=-1)+1e-5
    I = torch.eye(3, device=x.device)
    # make L as matrix full of one
    L = torch.ones(3, device=x.device)
    L = L.expand([x.shape[0], x.shape[1], 3, 3])

    for i in range(3):
        P = torch.clone(torch.eye(3, device=x.device).expand(
            [x.shape[0], x.shape[1], 3, 3]))
        P[:, :, i, i] *= x[..., i]
        L = P @ L @ P

    J = ln.reshape(x.shape[0], x.shape[1], 1, 1)*I.reshape(1, 1, 3, 3)

    J = (J-L)/(ln**(3/2)).reshape(x.shape[0], x.shape[1], 1, 1)

    return J/torch.sqrt(far.max())


def Jacobi_g(x, radius=3.):
    ln = 1./(torch.linalg.norm(x, axis=-1)+1e-5)

    I = torch.eye(3, device=x.device)
    L = torch.ones(3, device=x.device)
    L = L.expand([x.shape[0], x.shape[1], 3, 3])

    for i in range(3):
        P = torch.clone(torch.eye(3, device=x.device).expand(
            [x.shape[0], x.shape[1], 3, 3]))
        P[:, :, i, i] *= x[..., i]
        L = P @ L @ P
    P1 = (-radius*(ln**2)+2. *
          ln).reshape(x.shape[0], x.shape[1], 1, 1)*I.reshape(1, 1, 3, 3)
    # P2=2*(ln**3).unsqueeze(-1).bmm(L.reshape(L.shape[0],1,L.shape[1],9)).reshape(L.shape)
    P2 = (2. * radius*ln**4-2.*ln**3)[..., None, None].expand(L.shape)*L
#   P3=- 2.*(ln**3)[...,None,None]*x[...,None].expand(x.shape[0],x.shape[1],x.shape[2],3).permute(0,1,3,2)
    J1 = P1+P2
    l = (torch.linalg.norm(x, axis=-1)+1e-5)[..., None, None].expand(J1.shape)
    J = (l >= radius)*J1+(l < radius)*1./radius * \
        I.reshape(1, 1, 3, 3).expand(J1.shape)
    return J


def warp_fn(fn_idx, viewc, far, radius=3.):
    def fn1(x, viewc):
        return (x-viewc)/torch.sqrt((torch.linalg.norm(x-viewc, axis=-1)*far)[..., None])

    def fn2(x, radius=3.):
        l = (torch.linalg.norm(x, axis=-1)+1e-8)[..., None].expand(x.shape)
        r = (2.-radius/l)*x/l*(l > radius)+x/radius*(l <= radius)
        return r
    if fn_idx == 0:
        return[partial(fn1, viewc=viewc), partial(Jacobi_f, far=far)]
    else:
        return[partial(fn2, radius=radius), partial(Jacobi_g, radius=radius)]


def sample2enc(s_vals, origins, directions, radii, ray_shape, near, far, num_samples, fn_idx, viewc=0., radius=3, transform_idx=0):
    T = transform(transform_idx)
    t_vals = T(s_vals, near, far)
    N = t_vals.shape[0]
    means, covs = cast_rays(t_vals, origins, directions, radii, ray_shape)
    [fn, Jacobi_fn] = warp_fn(fn_idx, viewc=viewc, far=far, radius=3.)
    f_means = fn(means)
    jacobi = Jacobi_fn(means)
    # f_covs= jacobi @ torch.diag(covs) @jacobi.t()
    f_covs = torch.stack(
        [jacobi[..., :, i:i+1]@jacobi[..., i:i+1, :] for i in range(3)], 2)
    f_covs = f_covs.reshape(-1, 3, 3*3).permute(0, 2, 1).bmm(
        covs.reshape(N*num_samples, 3, 1)).reshape(N, num_samples, 3, 3)

    return (f_means, f_covs)


def transform(transform_idx=0):
    # sample for the transform style: log, disparity, linear
    if transform_idx == 0:
        return Transform_log
    elif transform_idx == 1:
        return Transform
    else:
        return Transform_linear
