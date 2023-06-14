from random import sample
from turtle import forward
import torch
from torch import nn
import model.mip as mip
from torch.nn import functional as F
import utils.render_utils as utils


class MipNerfModel(nn.Module):
    def __init__(self,
                 n_samples: int = 128,
                 n_levels: int = 2,
                 resample_padding: float = 0.01,
                 stop_level_grad: bool = True,
                 use_viewdirs: bool = True,
                 lindisp: bool = False,
                 ray_shape: str = "cylinder",
                 min_deg_point: int = 0,
                 max_deg_point: int = 16,
                 deg_view: int = 4,
                 density_noise: float = 1.,
                 density_bias: float = -1.,
                 rgb_padding: float = 0.001,
                 disable_integration: bool = False,
                 no_warp_sample=True,
                 fn=None,
                 radius=None,
                 real=False,
                 transform_idx=0,
                 rgb_layer=1,
                 hidden_layer=256,
                 encode_appearance=False,
                 N_vocab=100,
                 proposal_hidden_layer=256,
                 proposal_loss=False,
                 N_fine=128,
                 semantic=False,
                 semantic_class_num=0,):x
        super(MipNerfModel, self).__init__()

        self.n_levels = n_levels
        self.stop_level_grad = stop_level_grad
        self.deg_view = deg_view
        self.n_samples = n_samples
        self.lindisp = lindisp
        self.ray_shape = ray_shape
        self.resample_padding = resample_padding
        self.disable_integration = disable_integration
        self.min_deg_point = min_deg_point
        self.max_deg_point = max_deg_point
        self.use_viewdirs = use_viewdirs
        self.density_noise = density_noise
        self.rgb_padding = rgb_padding
        self.density_bias = density_bias
        self.no_warp_sample = no_warp_sample
        self.fn = fn
        self.mlp = MLP(feature_dim=self.max_deg_point*6, n_layers_condition=rgb_layer, n_units=hidden_layer, cond_dim=27+encode_appearance*48,
                       semantic=semantic, semantic_class_num=semantic_class_num,)
        self.radius = radius
        self.real = real
        self.transform_idx = transform_idx
        self.N_vocab = N_vocab
        self.encode_appearance = encode_appearance
        if encode_appearance:
            self.emb = torch.nn.Embedding(self.N_vocab, 48)
        self.proposal = proposal(
            n_units=proposal_hidden_layer, feature_dim=self.max_deg_point*6)
        self.proposal_loss = proposal_loss
        self.semantic = semantic
        self.N_fine = N_fine

    def forward(self, rays, randomized, white_bg, viewc):

        device = rays.origins.device

        ret = []
        for i_level in range(self.n_levels):

            if i_level == 0:
                # stratified sampling along rays
                if self.no_warp_sample:
                    t_vals, samples = mip.sample_along_rays(
                        rays.origins,
                        rays.directions,
                        rays.radii,
                        self.n_samples,
                        rays.near,
                        rays.far,
                        randomized,
                        self.lindisp,
                        self.ray_shape
                    )
                else:
                    s_vals, samples = mip.warp_sample_along_rays(
                        rays.origins,
                        rays.directions,
                        rays.radii,
                        self.n_samples,
                        rays.near,
                        rays.far,
                        randomized,
                        self.lindisp,
                        self.ray_shape, viewc=viewc, fn_idx=self.fn, radius=self.radius, transform_idx=self.transform_idx)
            else:
                if self.no_warp_sample:
                    t_vals, samples = mip.resample_along_rays(
                        rays.origins,
                        rays.directions,
                        rays.radii,
                        t_vals,
                        weights,
                        randomized,

                        self.ray_shape,
                        self.stop_level_grad,
                        resample_padding=self.resample_padding,

                    )
                else:
                    s_vals, samples = mip.warp_resample_along_rays(
                        rays.origins,
                        rays.directions,
                        rays.radii,
                        s_vals,
                        weights,
                        randomized,
                        self.N_fine,
                        self.ray_shape,
                        self.stop_level_grad,
                        resample_padding=self.resample_padding, viewc=viewc, near=rays.near, far=rays.far, fn_idx=self.fn, radius=self.radius, transform_idx=self.transform_idx
                    )
            if self.disable_integration:
                samples = (samples[0], torch.zeros_like(samples[1]))

            samples_enc = mip.integrated_pos_enc(
                samples,
                self.min_deg_point,
                self.max_deg_point,
                diag=self.no_warp_sample,
                device=device
            )
            raw_semantic = None
            if i_level == 0:
                raw_rgb, raw_density = self.proposal(samples_enc)
            else:
                if self.use_viewdirs:
                    viewdirs_enc = mip.pos_enc(
                        rays.viewdirs,
                        min_deg=0,
                        max_deg=self.deg_view,
                        append_identity=True,
                    )
                    if self.encode_appearance:
                        app_enc = self.emb(rays.app.long())
                        condition_enc = torch.cat(
                            [viewdirs_enc, app_enc.squeeze(1)], axis=1)

                        raw_rgb, raw_density, raw_semantic = self.mlp(
                            samples_enc, condition_enc)
                    else:
                        raw_rgb, raw_density, raw_semantic = self.mlp(
                            samples_enc, viewdirs_enc)
                else:
                    raw_rgb, raw_density = self.mlp(samples_enc)
            # print('raw_rgb',i_level,raw_rgb)
            if randomized and self.density_noise > 0:
                raw_density += self.density_noise * torch.randn(
                    *(raw_density.shape), dtype=raw_density.dtype, device=raw_density.device)
            if raw_rgb is not None:
                rgb = torch.sigmoid(raw_rgb)
                rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            else:
                rgb = None

            density = F.softplus(raw_density + self.density_bias)

            comp_rgb, distance, acc, weights, semantic = mip.real_volumetric_rendering(
                rgb, density, s_vals, rays.directions, raw_semantic, white_bkgd=white_bg, near=rays.near, far=rays.far, transform_idx=self.transform_idx)
            output = [comp_rgb, distance, acc]
            # if self.semantic:
            if i_level == 1:
                output.append(semantic)
            if self.proposal_loss:
                output += [s_vals, weights]
            ret.append(output)

        return ret


def make_mipnerf(args, device):
    model = MipNerfModel(no_warp_sample=args.no_warp_sample, disable_integration=args.disable_integration, ray_shape=args.ray_shape,
                         fn=args.fn, max_deg_point=args.max_degree, radius=args.radius, transform_idx=args.transform_idx, real=args.real, rgb_layer=args.rgb_layer,
                         hidden_layer=args.hidden_layer, density_noise=args.density_noise, encode_appearance=args.encode_appearance, n_samples=args.N_samples,
                         proposal_loss=args.proposal_loss, N_fine=args.N_fine, semantic=args.semantic, semantic_class_num=args.semantic_class_num)
    # model(example_rays, randomized, white_bg)

    return model


class DenseBlock(nn.Module):
    def __init__(self, input_channnels, output_channels: int = 256):
        super(DenseBlock, self).__init__()
        layers = [
            # use glorot uniform init
            nn.Linear(input_channnels, output_channels),
            nn.ReLU(inplace=True),
        ]
        torch.nn.init.xavier_uniform_(layers[0].weight)
        # torch.nn.init.constant_(layers[0].weight,0.01)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLP(nn.Module):
    def __init__(self,
                 n_layers: int = 8,
                 n_units: int = 256,
                 n_layers_condition: int = 1,
                 n_units_condition: int = 128,
                 skip_layer: int = 4,
                 n_rgb_channels: int = 3,
                 n_density_channels: int = 1,
                 feature_dim: int = 96,
                 cond_dim: int = 27,
                 condition=None,
                 semantic=False,
                 semantic_class_num=0,):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.skip_layer = skip_layer
        self.n_density_channels = n_density_channels
        self.n_rgb_channels = n_rgb_channels
        self.layers = nn.ModuleList()
        self.layers.append(DenseBlock(feature_dim, n_units))
        for i in range(n_layers-1):
            if i % self.skip_layer == 0 and i > 0:
                self.layers.append(DenseBlock(feature_dim+n_units, n_units))
            else:
                self.layers.append(DenseBlock(n_units, n_units))
        self.density_layer = nn.Linear(n_units, n_density_channels)
        # if condition is not None:
        self.bottleneck_layer = DenseBlock(n_units, n_units)
        self.cond_layers = nn.ModuleList()
        for i in range(n_layers_condition):
            if i == 0:
                self.cond_layers.append(DenseBlock(
                    n_units+cond_dim, n_units_condition))
            else:
                self.cond_layers.append(DenseBlock(
                    n_units_condition, n_units_condition))
        self.cond_layers = nn.Sequential(*self.cond_layers)
        self.rgb_layer = nn.Linear(n_units_condition, n_rgb_channels)
        self.semantic = semantic
        self.n_semantic_channels = semantic_class_num
        if semantic:
            self.semantic_layer = nn.Sequential(DenseBlock(
                n_units, n_units // 2), nn.Linear(n_units // 2, semantic_class_num))
            # torch.nn.init.xavier_uniform_(self.semantic_layer.weight)
        torch.nn.init.xavier_uniform_(self.density_layer.weight)
        torch.nn.init.xavier_uniform_(self.rgb_layer.weight)

    def forward(self, x, condition):
        feature_dim = x.size(-1)
        n_samples = x.size(1)
        x = x.reshape(-1, feature_dim)
        inputs = x
        for i in range(self.n_layers):
            # print('current_shape',x.shape)
            # print('current_x',x)
            x = self.layers[i](x)
            if i % self.skip_layer == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        raw_density = self.density_layer(x).reshape(
            -1, n_samples, self.n_density_channels)
        # print('density_x',x)
        raw_semantic = None
        if self.semantic:
            raw_semantic = self.semantic_layer(x).reshape(
                -1, n_samples, self.n_semantic_channels)
        if condition is not None:
            bottleneck = self.bottleneck_layer(x)
            condition = torch.tile(condition[:, None, :],
                                   (1, n_samples, 1))
            condition = condition.reshape([-1, condition.size(-1)])
            x = torch.cat([bottleneck, condition], dim=-1)
            x = self.cond_layers(x)

        raw_rgb = self.rgb_layer(x).reshape(
            -1, n_samples, self.n_rgb_channels)

        # print('raw_rgb0',raw_rgb[0,0,:])
        # print('raw_density0',raw_density[0,0,:])
        return raw_rgb, raw_density, raw_semantic


class proposal(nn.Module):
    def __init__(self,
                 n_units=256,
                 n_layers: int = 4,
                 n_density_channels: int = 1,
                 feature_dim: int = 96,
                 ):
        super(proposal, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.layers.append(DenseBlock(feature_dim, n_units))
        for i in range(n_layers-1):
            self.layers.append(DenseBlock(n_units, n_units))
        self.density_layer = nn.Linear(n_units, n_density_channels)
        self.n_density_channels = n_density_channels

    def forward(self, x,):
        feature_dim = x.size(-1)
        n_samples = x.size(1)
        x = x.reshape(-1, feature_dim)
        inputs = x
        for i in range(self.n_layers):
            x = self.layers[i](x)
        raw_density = self.density_layer(x).reshape(
            -1, n_samples, self.n_density_channels)
        raw_rgb = None
        return raw_rgb, raw_density


def render_image(render_fn, rays, rank, chunk=8192):
    n_devices = torch.cuda.device_count()  # one device for render
    height, width = rays[0].shape[:2]
    num_rays = height * width
    rays = utils.namedtuple_map(lambda r: r.reshape((num_rays, -1)), rays)
    results = []

    for i in range(0, num_rays, chunk):
        # pylint: disable=cell-var-from-loop
        chunk_rays = utils.namedtuple_map(lambda r: r[i:i + chunk], rays)
        chunk_size = chunk_rays[0].shape[0]
        rays_remaining = chunk_size % torch.cuda.device_count()
        if rays_remaining != 0:
            padding = n_devices - rays_remaining
            chunk_rays = utils.namedtuple_map(
                # mode = "edge", not reflect
                lambda r: F.pad(r, (0, padding, 0, 0), mode='reflect'), chunk_rays)
        else:
            padding = 0
        # After padding the number of chunk_rays is always divisible by
        # host_count.
        rays_per_host = chunk_rays[0].shape[0]
        start, stop = 0 * rays_per_host, (0 + 1) * rays_per_host
        chunk_results = render_fn(chunk_rays)[-1]
        results.append([utils.unshard(x[None, ...], padding)
                       for x in chunk_results])
    rgb, distance, acc, semantic_logits = [
        torch.cat(r, axis=0) for r in zip(*results)]
    rgb = rgb.reshape((height, width, -1))
    distance = distance.reshape((height, width))
    acc = acc.reshape((height, width))
    semantic = semantic_logits.reshape((height, width, -1))
    return (rgb, distance, acc, semantic)
