import imageio
import torch
import numpy as np
import os
import cv2

from .run_nerf_helpers import *
from .render import render_path, render_test_ray

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def render_only(args, render_kwargs_test, basedir, expname, render_depends, train_depends, splits, start):
    images, poses, intrinsics, depth_gts, bboxes = train_depends
    render_poses, render_depth, render_focal = render_depends
    images = torch.Tensor(images).to(_DEVICE)
    poses = torch.Tensor(poses).to(_DEVICE)
    render_poses = torch.Tensor(render_poses).to(_DEVICE)
    H, W = images[0].shape[:2]
    i_train, i_val, i_test = splits

    if args.render_test:
        # render_test switches to test poses
        images = images[i_test]
    else:
        # Default is smoother render_poses path
        images = None

    if args.render_test:
        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test', start))
    elif args.render_train:
        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('train', start))
    else:
        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('path', start))
    os.makedirs(testsavedir, exist_ok=True)
    print('test poses shape', render_poses.shape)

    if args.render_test_ray:
        index_pose = i_train[0]
        rays_o, rays_d = get_rays_by_coord_np(poses[index_pose,:3,:4], depth_gts[index_pose]['coord'], intrinsics[index_pose])
        rays_o, rays_d = torch.Tensor(rays_o).to(_DEVICE), torch.Tensor(rays_d).to(_DEVICE)
        rgb, sigma, z_vals, depth_maps, weights = render_test_ray(rays_o, rays_d, hwf, network=render_kwargs_test['network_fine'], **render_kwargs_test)

        for k in range(20):
            visualize_weights(weights[k*100, :].cpu().numpy(), z_vals[k*100, :].cpu().numpy(), os.path.join(testsavedir, f'rays_weights_%d.png' % k))
        print("colmap depth:", depth_gts[index_pose]['depth'][0])
        print("Estimated depth:", depth_maps[0].cpu().numpy())
        print(depth_gts[index_pose]['coord'])
    else:
        if(args.render_factor and type(render_depth) != type(None)):
            render_masks = []
            for dep_img in render_depth:
                img = cv2.resize(dep_img, (int(W/args.render_factor), int(H/args.render_factor)))
                render_mask = img==-1
                render_masks.append(render_mask)

        if(args.render_train):
            rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
        else:
            render_hwf = [H, W, render_focal]
            rgbs, disps = render_path(render_poses, render_hwf, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor, render_masks=render_masks)

        print('Done rendering', testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, 'rgb.mp4'), to8b(rgbs), fps=30, quality=8)
        disps[np.isnan(disps)] = 0
        print('Depth stats', np.mean(disps), np.max(disps), np.percentile(disps, 95))
        imageio.mimwrite(os.path.join(testsavedir, 'disp.mp4'), to8b(disps / np.percentile(disps, 95)), fps=30, quality=8)