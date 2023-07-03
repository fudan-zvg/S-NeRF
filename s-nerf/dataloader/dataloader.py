import os
import torch
import imageio
import numpy as np
from .load_llff import load_llff_data, load_colmap_depth
from .load_nuscenes import load_nuscenes_data, load_depth_map,load_flow,load_semantic


def load_llff(args):
    depth_gts, bboxes, ori_points, render_depth, render_focal = [None]*5

    if args.colmap_depth:
        depth_gts = load_colmap_depth(args.datadir, factor=args.factor, bd_factor=.75)
    images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                recenter=True, bd_factor=.75,
                                                                spherify=args.spherify)
    if args.block_bg:
        bbox_path = os.path.join(args.datadir, 'bboxes', f"{args.car_sample_n}_bboxes.pt")
        bboxes = torch.load(bbox_path).cpu().numpy()
        
    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
    print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
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

    print('DEFINING BOUNDS')
    if args.no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
        
    else:
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)

    train_depends = [images, hwf, poses, ori_points, depth_gts, bboxes]
    render_depends = [render_poses, render_depth, render_focal]
    splits = [i_train, i_val, i_test]

    return train_depends, bds, render_depends, splits

def load_nuscenes(args):
    i_test = []
    flow = None
    skymask = None
    semantic = None
    seg_masks = None
    H, W = args.H, args.W
    depth_gts, bds, bds_raw, skymask = load_depth_map(args.datadir,H,W,bd_factor=args.bds_factor,sky_mask=args.skymask)
    
    if args.flow:
        assert args.bds_factor == 0
        flow = load_flow(args.datadir)
    if args.semantic:
        semantic = load_semantic(args.datadir)
        semantic_index, semantic_labels = semantic

    if args.seg_mask:
        seg_mask_dir=os.path.join(args.datadir,'seg')
        img_files = sorted(os.listdir(seg_mask_dir), key=lambda x : int(x.split('.')[0]))
        imgfiles = [os.path.join(seg_mask_dir, f) for f in img_files if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

        imgs = [imageio.imread(f).astype(np.float32)/255. for f in imgfiles]
        seg_masks = np.stack(imgs, 0)
        seg_masks = torch.tensor(seg_masks)

    images, poses, render_poses, render_depth, render_focal, K = load_nuscenes_data(args, bds_raw,bd_factor=args.bds_factor)       
    print('Loaded Nuscenes', images.shape, poses.shape, args.datadir)
    if args.load_poses is not None:
        poses = np.load(os.path.join(args.load_poses,'poses_bounds.npy'))

    if args.cam_num:
        cam_index=[i for i in range(args.cam_num)for j in range(images.shape[0]//args.cam_num) ]
    else:
        cam_index=[0]*images.shape[0]    
    
    train_depends = [images, poses, K, depth_gts, flow, cam_index, skymask, seg_masks, semantic]
    
    i_val = i_test = [i for i in range(images.shape[0])][::args.datahold]
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                     (i not in i_test and i not in i_val)])
    
    if args.render_test:
        render_poses = np.array(poses[i_test])
    elif args.render_train:
        render_poses = np.array(poses[i_train])
    render_depends = [render_poses, render_depth, render_focal]

    if args.half_train:
        i_train=i_train[::2]
    elif args.fulltrain:
        i_train = np.array([i for i in range(images.shape[0])])
    
    if args.semantic:
        i_train = np.concatenate([i_train, semantic_index])
        i_train = np.unique(i_train) 

    splits = [i_train, i_val, i_test]

    return train_depends, bds_raw, render_depends, splits


def load_dataset(args):
    if args.dataset_type == 'llff':
        return load_llff(args)
    elif args.dataset_type == 'nuscenes':
        return load_nuscenes(args)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        exit(-1)
