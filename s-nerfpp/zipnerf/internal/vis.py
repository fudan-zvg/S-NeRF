from internal import stepfun
import numpy as np
from matplotlib import cm


def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    if len(x.shape) != len(w.shape):
        w = np.broadcast_to(w[..., None], x.shape)
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)


def sinebow(h):
    """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
    f = lambda x: np.sin(np.pi * x) ** 2
    return np.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def matte(vis, acc, dark=0.8, light=1.0, width=8):
    """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    bg_mask = np.logical_xor(
        (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
        (np.arange(acc.shape[1]) % (2 * width) // width)[None, :])
    bg = np.where(bg_mask, light, dark)
    return vis * acc[:, :, None] + (bg * (1 - acc))[:, :, None]


def visualize_cmap(value,
                   weight,
                   colormap,
                   lo=None,
                   hi=None,
                   percentile=99.,
                   curve_fn=lambda x: x,
                   modulus=None,
                   matte_background=True):
    """Visualize a 1D image and a 1D weighting according to some colormap.

  Args:
    value: A 1D image.
    weight: A weight map, in [0, 1].
    colormap: A colormap function.
    lo: The lower bound to use when rendering, if None then use a percentile.
    hi: The upper bound to use when rendering, if None then use a percentile.
    percentile: What percentile of the value map to crop to when automatically
      generating `lo` and `hi`. Depends on `weight` as well as `value'.
    curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
      before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
    modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
      `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
    matte_background: If True, matte the image over a checkerboard.

  Returns:
    A colormap rendering.
  """
    # Identify the values that bound the middle of `value' according to `weight`.
    lo_auto, hi_auto = weighted_percentile(
        value, weight, [50 - percentile / 2, 50 + percentile / 2], assume_sorted=True)

    # If `lo` or `hi` are None, use the automatically-computed bounds above.
    eps = np.finfo(np.float32).eps
    lo = lo or (lo_auto - eps)
    hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
            np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))

    if colormap:
        colorized = colormap(value)[:, :, :3]
    else:
        if len(value.shape) != 3:
            raise ValueError(f'value must have 3 dims but has {len(value.shape)}')
        if value.shape[-1] != 3:
            raise ValueError(
                f'value must have 3 channels but has {len(value.shape[-1])}')
        colorized = value

    return matte(colorized, weight) if matte_background else colorized


def visualize_coord_mod(coords, acc):
    """Visualize the coordinate of each point within its "cell"."""
    return matte(((coords + 1) % 2) / 2, acc)


def visualize_rays(dist,
                   dist_range,
                   weights,
                   rgbs,
                   accumulate=False,
                   renormalize=False,
                   resolution=2048,
                   bg_color=0.8):
    """Visualize a bundle of rays."""
    dist_vis = np.linspace(*dist_range, resolution + 1)
    vis_rgb, vis_alpha = [], []
    for ds, ws, rs in zip(dist, weights, rgbs):
        vis_rs, vis_ws = [], []
        for d, w, r in zip(ds, ws, rs):
            if accumulate:
                # Produce the accumulated color and weight at each point along the ray.
                w_csum = np.cumsum(w, axis=0)
                rw_csum = np.cumsum((r * w[:, None]), axis=0)
                eps = np.finfo(np.float32).eps
                r, w = (rw_csum + eps) / (w_csum[:, None] + 2 * eps), w_csum
            vis_rs.append(stepfun.resample_np(dist_vis, d, r.T, use_avg=True).T)
            vis_ws.append(stepfun.resample_np(dist_vis, d, w.T, use_avg=True).T)
        vis_rgb.append(np.stack(vis_rs))
        vis_alpha.append(np.stack(vis_ws))
    vis_rgb = np.stack(vis_rgb, axis=1)
    vis_alpha = np.stack(vis_alpha, axis=1)

    if renormalize:
        # Scale the alphas so that the largest value is 1, for visualization.
        vis_alpha /= np.maximum(np.finfo(np.float32).eps, np.max(vis_alpha))

    if resolution > vis_rgb.shape[0]:
        rep = resolution // (vis_rgb.shape[0] * vis_rgb.shape[1] + 1)
        stride = rep * vis_rgb.shape[1]

        vis_rgb = np.tile(vis_rgb, (1, 1, rep, 1)).reshape((-1,) + vis_rgb.shape[2:])
        vis_alpha = np.tile(vis_alpha, (1, 1, rep)).reshape((-1,) + vis_alpha.shape[2:])

        # Add a strip of background pixels after each set of levels of rays.
        vis_rgb = vis_rgb.reshape((-1, stride) + vis_rgb.shape[1:])
        vis_alpha = vis_alpha.reshape((-1, stride) + vis_alpha.shape[1:])
        vis_rgb = np.concatenate([vis_rgb, np.zeros_like(vis_rgb[:, :1])],
                                  axis=1).reshape((-1,) + vis_rgb.shape[2:])
        vis_alpha = np.concatenate(
            [vis_alpha, np.zeros_like(vis_alpha[:, :1])],
            axis=1).reshape((-1,) + vis_alpha.shape[2:])

    # Matte the RGB image over the background.
    vis = vis_rgb * vis_alpha[..., None] + (bg_color * (1 - vis_alpha))[..., None]

    # Remove the final row of background pixels.
    vis = vis[:-1]
    vis_alpha = vis_alpha[:-1]
    return vis, vis_alpha


num_class = 19
def def_color_map():
    s = 256**3//num_class
    colormap = [[(i*s)//(256**2),((i*s)//(256)%256),(i*s)%(256) ] for i in range(num_class)]
    return colormap
color_map = np.array(def_color_map())

def visualize_depth(depth, near=0.2, far=13):
    colormap = cm.get_cmap('turbo')
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]
    # os.makedirs('./test_depths/'+pathname,exist_ok=True)
    out_depth = np.clip(np.nan_to_num(vis), 0., 1.) * 255
    return out_depth

def visualize_suite(rendering, batch):
    """A wrapper around other visualizations for easy integration."""

    # depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)

    rgb = rendering['rgb']
    acc = rendering['acc']

    distance_mean = rendering['distance_mean']

    logits_2_label = lambda x: np.argmax(x, axis=-1)
    labels = logits_2_label(rendering['semantic'])
    labels = color_map[labels]/255
    dep = visualize_depth(rendering['depth'])/255
    # distance_median = rendering['distance_median']
    # distance_p5 = rendering['distance_percentile_5']
    # distance_p95 = rendering['distance_percentile_95']
    acc = np.where(np.isnan(distance_mean), np.zeros_like(acc), acc)

    # The xyz coordinates where rays terminate.
    coords = batch['origins'] + batch['directions'] * distance_mean[:, :, None]

    # vis_depth_mean, vis_depth_median = [
    #     visualize_cmap(x, acc, cm.get_cmap('turbo'), curve_fn=depth_curve_fn)
    #     for x in [distance_mean, distance_median]
    # ]

    # Render three depth percentiles directly to RGB channels, where the spacing
    # determines the color. delta == big change, epsilon = small change.
    #   Gray: A strong discontinuitiy, [x-epsilon, x, x+epsilon]
    #   Purple: A thin but even density, [x-delta, x, x+delta]
    #   Red: A thin density, then a thick density, [x-delta, x, x+epsilon]
    #   Blue: A thick density, then a thin density, [x-epsilon, x, x+delta]
    # vis_depth_triplet = visualize_cmap(
    #     np.stack(
    #         [2 * distance_median - distance_p5, distance_median, distance_p95],
    #         axis=-1),
    #     acc,
    #     None,
    #     curve_fn=lambda x: np.log(x + np.finfo(np.float32).eps))

    # dist = rendering['ray_sdist']
    # dist_range = (0, 1)
    # weights = rendering['ray_weights']
    # rgbs = [np.clip(r, 0, 1) for r in rendering['ray_rgbs']]
    #
    # vis_ray_colors, _ = visualize_rays(dist, dist_range, weights, rgbs)
    #
    # sqrt_weights = [np.sqrt(w) for w in weights]
    # sqrt_ray_weights, ray_alpha = visualize_rays(
    #     dist,
    #     dist_range,
    #     [np.ones_like(lw) for lw in sqrt_weights],
    #     [lw[..., None] for lw in sqrt_weights],
    #     bg_color=0,
    # )
    # sqrt_ray_weights = sqrt_ray_weights[..., 0]
    #
    # null_color = np.array([1., 0., 0.])
    # vis_ray_weights = np.where(
    #     ray_alpha[:, :, None] == 0,
    #     null_color[None, None],
    #     visualize_cmap(
    #         sqrt_ray_weights,
    #         np.ones_like(sqrt_ray_weights),
    #         cm.get_cmap('gray'),
    #         lo=0,
    #         hi=1,
    #         matte_background=False,
    #     ),
    # )

    vis = {
        'color': rgb,
        'acc': acc,
        'color_matte': matte(rgb, acc),
        'depth': dep,
        'semantic': labels,
        # 'depth_mean': vis_depth_mean,
        # 'depth_median': vis_depth_median,
        # 'depth_triplet': vis_depth_triplet,
        'coords_mod': visualize_coord_mod(coords, acc),
        # 'ray_colors': vis_ray_colors,
        # 'ray_weights': vis_ray_weights,
    }

    if 'rgb_cc' in rendering:
        vis['color_corrected'] = rendering['rgb_cc']

    # Render every item named "normals*".
    for key, val in rendering.items():
        if key.startswith('normals'):
            vis[key] = matte(val / 2. + 0.5, acc)

    if 'roughness' in rendering:
        vis['roughness'] = matte(np.tanh(rendering['roughness']), acc)

    return vis


def render_test_semantic(sum_wrt, batch, step, name='test_true_semantic'):
    logits_2_label = lambda x: np.argmax(x, axis=-1)
    labels = logits_2_label(batch['semantic'])
    labels = color_map[labels]/255
    sum_wrt.add_image(name, labels, step)
