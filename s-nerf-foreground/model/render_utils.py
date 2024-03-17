import imageio
import torch
import numpy as np
import os

from .run_nerf_helpers import *
from .render import render_path, render_test_ray
from utils.test_utils import calc_bbox_benchmark, calc_mask_psnr

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def render_only(args, render_kwargs_test, basedir, expname, render_depends, train_depends, pose_param_net, splits, start):
    images, poses, intrinsics, depth_gts, depth_ori, bboxes, _ = train_depends
    render_poses, render_depth, render_focal, render_dist = render_depends
    images = torch.Tensor(images).to(_DEVICE)
    poses = torch.Tensor(poses).to(_DEVICE)
    render_poses = torch.Tensor(render_poses).to(_DEVICE)
    H, W = images[0].shape[:2]
    i_train, i_val, i_test = splits

    if args.render_train:
        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('train', start))
    else:
        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('path', start))
    os.makedirs(testsavedir, exist_ok=True)
    print('render poses shape', render_poses.shape)

    if(args.render_train):
        if(args.pose_refine):
            render_poses = [pose_param_net(i) for i in i_train]
            render_poses = torch.stack(render_poses)
        rgbs, deps = render_path(H, W, render_poses, intrinsics[i_train], args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=1)
        psnr = calc_bbox_benchmark(images[i_train], rgbs, bboxes[i_train])
        for idx, img in enumerate(rgbs):
            x1,y1,x2,y2 = bboxes[i_train][idx].round().astype(int)
            img_crop = img[y1:y2, x1:x2]
            imageio.imwrite(os.path.join(testsavedir, f'img{idx}_{psnr[idx].item()}.png'), img_crop)
        
        print(f"Train_PSNR: {psnr}")
    else:
        render_focal = render_focal.item()
        render_intr = torch.tensor([[[render_focal, 0, W/2],
                                    [0, render_focal, H/2],
                                    [0, 0, 1]]])
        rgbs, deps = render_path(H, W, render_poses, render_intr, args.chunk, render_kwargs_test, 
                                gt_imgs=images, 
                                savedir=testsavedir, 
                                render_factor=args.render_factor, 
                                render_depth=render_depth,
                                render_dist=render_dist)

    # for idx, rgb in enumerate(rgbs):
    #     imageio.imwrite(os.path.join(testsavedir, f'{idx}.png'), to8b(rgb))
    print('Done rendering', testsavedir)
    imageio.mimwrite(os.path.join(testsavedir, 'rgb.mp4'), to8b(rgbs), fps=30, quality=8)
    deps[np.isnan(deps)] = 0
    imageio.mimwrite(os.path.join(testsavedir, 'disp.mp4'), deps, fps=30, quality=8)