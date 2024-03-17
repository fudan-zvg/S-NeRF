import numpy as np
import torch
import os, imageio
from pathlib import Path
import random
import copy
import pickle
from scipy.spatial.transform import Rotation as R
from utils.generate_renderpath import generate_render_path, recenter_poses


def preprocess_poses(raw_pose):
    P = raw_pose.copy()
    
    # rot = P[3, :3,:3]
    # P[:, :3,:3] = P[:, :3,:3] @ rot.T
    poses = np.concatenate([P[:, :, 1:2],P[:, :, 0:1], -P[:, :, 2:3], P[:, :, 3:4]], 2) # convert [r, -u, t] to [-u, r, -t]
    poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)  # convert [-u, r, -t] to [r, u, -t]
    # R.from_matrix(poses[2,:3,:3]).as_euler('yxz')*180/np.pi
    return poses

def load_imgs(path, car_n, mask=False):
    if(mask):
        imgdir = os.path.join(path, 'mask_images', str(car_n))
    else:
        imgdir = os.path.join(path, 'images', str(car_n))
    img_files = sorted(os.listdir(imgdir), key=lambda x : int(x.split('.')[0]))
    imgfiles = [os.path.join(imgdir, f) for f in img_files if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f) # , ignoregamma=True
        else:
            return imageio.imread(f)

    imgs = [imread(f)[...,:]/255. for f in imgfiles] # If white bkgd, load all 4 channels
    imgs = np.stack(imgs, 0)
    return imgs

def load_poses(path, car_n, N, sc, recenter=True):
    P_path = [os.path.join(path, f"cam_pose/{car_n}", f"P_{i}.pt") for i in range(N)]
    K_path = [os.path.join(path, f"cam_pose/{car_n}", f"K_{i}.pt") for i in range(N)]
    P = [torch.load(P_).cpu().numpy() for P_ in P_path] 
    K = [torch.load(K_).cpu().numpy() for K_ in K_path]
    P = np.stack(P, 0)
    intrinsics = np.stack(K, 0)

    poses = preprocess_poses(P)
    poses[:, :3, 3] *= sc
    if(recenter):
        poses, c2w = recenter_poses(poses)
        return poses, intrinsics, c2w
    return poses, intrinsics, None

def load_render_depends(path, car_n, sc, H, W):
    if(not os.path.exists(os.path.join(path, f"render/{car_n}"))):
        return None

    raw_render_poses = np.load(os.path.join(path, f"render/{car_n}", "poses.npy"))
    render_focal = np.load(os.path.join(path, f"render/{car_n}", "focal.npy"))

    depth_path = os.path.join(path, f"render/{car_n}", "render_depth.pkl")
    with open(depth_path, 'rb') as f:
        render_depth = pickle.load(f)

    for i in range(len(render_depth)):
        render_depth[i]['depth'] *= sc
    render_depth = restore_dep_img(H, W, render_depth)
    render_poses = preprocess_poses(raw_render_poses)
    render_poses[:, :3, 3] *= sc

    render_depends = [render_poses, render_depth, render_focal]
    return render_depends

def load_bboxes(path, car_n):
    bbox_path = os.path.join(path, 'bboxes', f"{car_n}.pt")
    bboxes = torch.load(bbox_path).cpu().numpy()
    return bboxes

def load_depth_map(path, car_n, bd_factor):
    depth_file = os.path.join(path, f'depth/{car_n}.pkl')
    with open(depth_file, 'rb') as f:
        depth = pickle.load(f)

    depth_bds = [np.array([dep['depth'].min(), dep['depth'].max()]) for dep in depth]
    bds_raw = np.stack(depth_bds)
    sc = 1. if bd_factor is None else 1./(bds_raw.min() * bd_factor)
    bds = bds_raw * sc
    
    for i in range(len(depth)):
        depth[i]['depth'] *= sc
    
    return depth, bds, bds_raw, sc
    

def load_nuscenes_data(args, bd_factor=.75, ret_full=False, N_views=120):
    path, car_n= args.datadir, args.car_sample_n
    depth_ori = None
    raw_imgs = load_imgs(path, car_n, args.white_bkgd)
    # if(args.bg_depth):
    #     [imgs[:,:,:,i]==1 for i in range(len(imgs))]
    imgs = raw_imgs if not args.white_bkgd else raw_imgs[...,:3]*raw_imgs[...,-1:] + (1.-raw_imgs[...,-1:])
    H, W = imgs[0].shape[:2]

    depth, bds, bds_raw, sc = load_depth_map(path, car_n, bd_factor)
    
    poses, intrinsics, c2w = load_poses(path, car_n, imgs.shape[0], sc, args.recenter)
    bboxes = load_bboxes(path, car_n)

    # render_depends = load_render_depends(path, car_n, sc, H, W)

    focal = intrinsics[:,0,0].mean()
    i_train = np.array(args.train_scene)
    raw_poses = c2w @ poses if(args.recenter) else poses
    
    render_poses, focal, render_dis = generate_render_path(raw_poses[i_train], focal, sc, N_views)
    render_depends = [render_poses, [], focal, render_dis]

    if(args.recenter and render_depends):
        render_poses = render_depends[0]
        render_poses = np.linalg.inv(c2w) @ render_poses
        render_depends[0] = render_poses

    if(ret_full):
        depth_ori = restore_dep_img(H, W, depth)

    train_depends = [imgs, poses, intrinsics, depth, depth_ori, bboxes, sc]

    return train_depends, render_depends, bds, raw_imgs


def restore_dep_img(H, W, depth):
    depth_images = []
    for dep_info in depth:
        coord = dep_info['coord']
        dep_val = dep_info['depth']
        depth_image = np.zeros((H, W, 1))
        depth_image[coord[:, 1], coord[:, 0]] = dep_val
        depth_images.append(depth_image)
    depth_images = np.stack(depth_images)
    return depth_images