import os
import torch
import numpy as np
from .load_llff import load_llff_data, load_colmap_depth
from .load_nuscenes import load_nuscenes_data, load_depth_map
from utils.generate_renderpath import generate_render_path


def load_llff(args):
    depth_gts, bboxes, ori_points, render_depth, render_focal = [None]*5

    if args.colmap_depth:
        depth_gts = load_colmap_depth(args.datadir, factor=args.factor, bd_factor=.75)
    images, poses, bds, raw_poses, i_test, sc = load_llff_data(args.datadir, args.factor,
                                                                recenter=True, bd_factor=.75,
                                                                spherify=args.spherify)
    if args.block_bg:
        bbox_path = os.path.join(args.datadir, 'bboxes', f"{args.car_sample_n}_bboxes.pt")
        bboxes = torch.load(bbox_path).cpu().numpy()
    
    hwf = poses[0,:3,-1]
    H, W, focal = hwf
    poses = poses[:,:3,:4]
    print('Loaded llff', images.shape, raw_poses.shape, hwf, args.datadir)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if args.llffhold > 0:
        print('Auto LLFF holdout,', args.llffhold)
        i_test = np.arange(images.shape[0])[::args.llffhold]

    if args.test_scene is not None:
        i_test = np.array([i for i in args.test_scene])

    if i_test[0] < 0:
        i_test = []

    i_val = i_test
    if args.train_scene is None:
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                    (i not in i_test and i not in i_val)])
    else:
        i_train = np.array([i for i in args.train_scene if
                    (i not in i_test and i not in i_val)])

    render_poses, focal, render_dis = generate_render_path(raw_poses[i_train], focal, sc, 10)
    render_depends = [render_poses, [], focal, render_dis]

    print('DEFINING BOUNDS')
    if args.no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
        
    else:
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)
    intrinsics = np.array([[[focal, 0, W//2],
                            [0, focal, H//2],
                            [0, 0,     1]]]*len(images))

    train_depends = [images, poses, intrinsics, depth_gts, None, None, None]
    splits = [i_train, i_val, i_test]

    return train_depends, bds, render_depends, splits, None

def load_nuscenes(args):
    i_test = []
    ret_full = False
    test_mask = None
    if(args.smooth_loss or args.psnr_method == 'mask'):
        ret_full = True
    train_depends, render_depends, bds, raw_imgs = load_nuscenes_data(args, ret_full=ret_full)      
    imgs = train_depends[0]
    print('Loaded Nuscenes')
    
    if args.test_scene is not None:
        i_test = np.array([i for i in args.test_scene])

    i_val = i_test
    if args.train_scene is None:
        i_train = np.array([i for i in np.arange(int(imgs.shape[0])) if
                    (i not in i_test and i not in i_val)])
    else:
        i_train = np.array([i for i in args.train_scene if
                    (i not in i_test and i not in i_val)])        
    
    splits = [i_train, i_val, i_test]

    if(args.white_bkgd):
        test_mask = raw_imgs[i_test][...,-1]
        test_mask = test_mask==1

    return train_depends, bds, render_depends, splits, test_mask


def load_dataset(args):
    if args.dataset_type == 'llff':
        return load_llff(args)
    elif args.dataset_type == 'nuscenes':
        return load_nuscenes(args)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        exit(-1)
