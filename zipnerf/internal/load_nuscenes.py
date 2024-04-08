import numpy as np
import torch
import os, imageio
from pathlib import Path
import random
import copy
from scipy.spatial.transform import Rotation as R
import cv2

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    c2w=c2w.astype(np.float32)
    return c2w

def recenter_poses(poses):
    
    poses_ = poses.copy()
    bottom = np.reshape([0,0,0,1.], [1,4])
    
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2).astype(np.float32)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = np.concatenate([poses_[:,:3,:4], bottom], -2).astype(np.float32)
    return poses, c2w

def generate_render_path(poses, bds):

    c2w = poses_avg(poses)
    print('recentered', c2w.shape)
    print('avg_c2w:',c2w[:3,:4])

    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min()*.9, bds.max()*2.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_views = 120
    N_rots = 2

    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    return render_poses

def preprocess_poses(raw_pose):
    P = raw_pose.copy()
    
    # rot = P[3, :3,:3]
    # P[:, :3,:3] = P[:, :3,:3] @ rot.T
    poses = np.concatenate([P[:, :, 1:2],P[:, :, 0:1], -P[:, :, 2:3], P[:, :, 3:4]], 2) # convert [r, -u, t] to [-u, r, -t]
    poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)  # convert [-u, r, -t] to [r, u, -t]
    # R.from_matrix(poses[2,:3,:3]).as_euler('yxz')*180/np.pi
    return poses


def load_waymo_meta(root_dir, factor=1):
    imgdir = os.path.join(root_dir, 'images')
    img_files = sorted(os.listdir(imgdir))
    img_files = [os.path.join(root_dir, 'images', img_file) for img_file in img_files]

    num = len(img_files)

    with open(os.path.join(root_dir, 'poses_bounds.npy'),
              'rb') as fp:
        poses_arr = np.load(fp).astype(np.float32)

    print('poses_arr', poses_arr.shape)
    poses = poses_arr[:, :-4].reshape([-1, 3, 5])
    bds = poses_arr[:, -4:-2].transpose([1, 0])
    raw_hw=poses_arr[:, -2:].transpose([1, 0])
    raw_hw = raw_hw.astype(int)
    raw_cam_K = poses[:, :, 4].copy().astype(np.float32).transpose([1, 0])
    cx=raw_cam_K[0,:]/factor
    cy=raw_cam_K[1,:]/factor
    focal=raw_cam_K[2,:]/factor
    K=[np.array([[focal[i],0,cx[i]],[0,focal[i],cy[i]],[0,0,1]]) for i in range(num)]
    K = np.stack(K, 0)

    poses = np.concatenate(
            [poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)
    return img_files, poses[:,:,:4], K, raw_hw

def load_png_semantic(mask_paths):
    masks = []
    for i, mask_path in enumerate(mask_paths):
        mask = imageio.imread(mask_path)
        if mask.shape[0] == 886:
            mask_z = np.zeros([1280,1920])
            mask_z[:886] = mask
            mask = mask_z
        masks.append(mask)
    return np.stack(masks, 0)


def load_nuscenes_data(root_dir, bds_raw, bd_factor=.75):
    render_focal = None
    render_depth = None
    path=root_dir
    sc = 1. if bd_factor==0. else 1./(bds_raw.min() * bd_factor)
    imgdir = os.path.join(path, 'images')
    img_files = sorted(os.listdir(imgdir), key=lambda x : int(x.split('.')[0]))
    imgfiles = [os.path.join(imgdir, f) for f in img_files if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    imgs = [imageio.imread(f)[...,:3].astype(np.float32)/255. for f in imgfiles]
    imgs = np.stack(imgs, 0)
    num = imgs.shape[0]
    # P_path = [os.path.join(path, f"cam_pose/{car_n}", f"P_{i}.pt") for i in range(num)]
    # K_path = [os.path.join(path, f"cam_pose/{car_n}", f"K_{i}.pt") for i in range(num)]
    # P = [torch.load(P_).cpu().numpy() for P_ in P_path] # world2cam
    # K = [torch.load(K_).cpu().numpy() for K_ in K_path]
    # P = np.stack(P, 0)
    # K = np.stack(K, 0)
    with open(os.path.join(root_dir, 'poses_bounds.npy'),
                                'rb') as fp:
            poses_arr = np.load(fp).astype(np.float32)
    
    print('poses_arr',poses_arr.shape)
    # print('Colmap:',args.colmap)
    ## colmap:n*17 nuscenes:n*19
    # if args.colmap:
    #     poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    #     bds = poses_arr[:, -2:].transpose([1, 0])
    #     raw_hw=poses[:,:2,4]
    # else:
    poses = poses_arr[:, :-4].reshape([-1, 3, 5])
    bds = poses_arr[:, -4:-2].transpose([1, 0])
    raw_hw=poses_arr[:, -2:].transpose([1, 0])
    if poses.shape[0] != imgs.shape[0]:
        raise RuntimeError('Mismatch between imgs {} and poses {}'.format(
            imgs.shape[-1], poses.shape[-1]))
    raw_cam_K=poses[:,:,4].copy().astype(np.float32).transpose([1,0]);
    factor=raw_hw[0,0]/imgs.shape[1]
    print('factor',factor)
    # if args.colmap:
    #     cx=raw_cam_K[1,:]/factor*.5
    #     cy=raw_cam_K[0,:]/factor*.5
    #     focal=raw_cam_K[2,:]/factor
    # else:
    cx=raw_cam_K[0,:]/factor
    cy=raw_cam_K[1,:]/factor
    focal=raw_cam_K[2,:]/factor
    K=[np.array([[focal[i],0,cx[i]],[0,focal[i],cy[i]],[0,0,1]]) for i in range(num)]
    K = np.stack(K, 0)

    # focal_list = [K_[0][0] for K_ in K]
    # ori_points = [[K_[0, 2], K_[1,2]] for K_ in K]
    
    # poses = preprocess_poses(P)
    poses = np.concatenate(
            [poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)
    
    poses[:, :3, 3] *= sc
    bds = bds_raw * sc
    poses, c2w = recenter_poses(poses)
    np.save(os.path.join(root_dir,'c2w_recenter.npy'),c2w)

    # TODO: accomplish the render poses and depth
    render_poses = generate_render_path(poses, bds)
    render_poses=np.array(render_poses)
    render_poses[:, :3, 3] *= sc

    render_depth = np.ones(len(render_poses))
    render_focal = np.ones(len(render_poses))
    
    return imgs, poses[:,:,:4], render_poses, render_depth, render_focal, K

def load_flow(path):
    path = os.path.join(path, 'flow')
    flows_path = sorted(os.listdir(path))
    next_flows, prev_flows = [], []
    for flow_path in flows_path[::2]:
        next_flow = np.load(os.path.join(path, flow_path))
        next_flows.append(next_flow)
    for flow_path in flows_path[1::2]:
        prev_flow = np.load(os.path.join(path, flow_path))
        prev_flows.append(prev_flow)

    next_flows = np.stack(next_flows)
    prev_flows = np.stack(prev_flows)
    flows = np.stack([next_flows, prev_flows])
    return flows

def load_semantic(path):
    semantic_path = os.path.join(path, 'semantic_labels.npy')
    try:
        semantic = np.load(semantic_path)
        semantic_labels = semantic[...,1]
        semantic_index = semantic[:,0,0,0,0]
        return semantic_index,semantic_labels
    except:
        return [],None
def load_full_semantic(path,segformer=False,label_specific = ''):
    if not segformer:
        return np.load(os.path.join(path,'full_semantic.npy'))
    else:
        if not label_specific:
            return np.load(os.path.join(path,'full_semantic_mmseg.npy')).astype(int64)
        else:
            return np.load(os.path.join(path,label_specific)).astype(int64)


def load_depth_map(path, H, W, bd_factor=.75, sky_mask=False):
    skymask = None
    depth_dir = os.path.join(path, 'depth')
    imgdir = depth_dir
    img_files = sorted(os.listdir(imgdir), key=lambda x: int(x.split('.')[0]))
    imgfiles = [os.path.join(imgdir, f) for f in img_files if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    # for f in imgfiles:
    #     if  cv2.imread(f, -1) is None:
    #         print(f)
    imgs = [cv2.resize(cv2.imread(f, -1) / 256., (W, H)) for f in imgfiles]
    depth = np.stack(imgs, 0).astype(np.float32)

    sky_depth = 100
    ## set the bounds for depth
    skymask = depth > 150
    depth[skymask] = sky_depth
    if sky_mask:
        skymask = depth > 150
    depth[depth > 0.5] = np.clip(depth[depth > 0.5], a_min=max(depth[depth > 0.5].min(), 2), a_max=sky_depth)

    #set near and far ignoring the sky, and set the sky as background
    fg_scale = 1.
    bds_raw = np.array([[max(dep[dep > 0.5].min(), 2.), dep[dep < 150].max()*fg_scale] if dep.sum()>0 else [2,100] for dep in depth ])

    sc = 1. if bd_factor == 0. else 1. / (bds_raw.min() * bd_factor)

    # Transpose Index to W, H
    # idx_dirs = [idx_dir[:,::-1] for idx_dir in idx_dirs]

    # near = np.ndarray.min(bds_raw) * .9 * sc
    # far = np.ndarray.max(bds_raw) * 1.2 * sc
    # near = np.ndarray.min(bds_raw) * .9 * sc
    # far = np.ndarray.max(bds_raw) * sc
    # print('near/far:', near, far)
    depth = depth * sc

    # for dep in depth_values:
    #     dep *= sc

    # depth_out = [{
    #     'depth': d_value,
    #     'coord': i_dir,
    #     'weight': np.ones_like(d_value)
    # } for d_value, i_dir in zip(depth_values, idx_dirs)]

    return depth, (depth[depth > 0.5].min(), depth[depth < 150].max()), bds_raw, skymask


# ## the ori function
# def load_depth_map(path, H,W,bd_factor=.75,sky_mask=False):
#     skymask=None
#     depth_dir = os.path.join(path,'depth')
#     imgdir=depth_dir
#     img_files = sorted(os.listdir(imgdir), key=lambda x : int(x.split('.')[0]))
#     imgfiles = [os.path.join(imgdir, f) for f in img_files if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
#
#     imgs = [cv2.resize(cv2.imread(f,-1)/256.,(W,H)) for f in imgfiles]
#     depth = np.stack(imgs, 0).astype(np.float32)
#     ## set the bounds for depth
#     if sky_mask:
#         skymask=depth>150
#     depth[depth>0.5] = np.clip(depth[depth>0.5],a_min=max(depth[depth>0.5].min(),2),a_max=100)
#
#     bds_raw = np.array([[max(dep[dep>0.5].min(),2.), dep[dep<150].max()] for dep in depth])
#
#     sc = 1. if bd_factor == 0. else 1./(bds_raw.min() * bd_factor)
#
#     # Transpose Index to W, H
#     # idx_dirs = [idx_dir[:,::-1] for idx_dir in idx_dirs]
#
#     # near = np.ndarray.min(bds_raw) * .9 * sc
#     # far = np.ndarray.max(bds_raw) * 1.2 * sc
#     # near = np.ndarray.min(bds_raw) * .9 * sc
#     # far = np.ndarray.max(bds_raw) * sc
#     # print('near/far:', near, far)
#     depth=depth*sc
#
#     # for dep in depth_values:
#     #     dep *= sc
#
#     # depth_out = [{
#     #     'depth': d_value,
#     #     'coord': i_dir,
#     #     'weight': np.ones_like(d_value)
#     # } for d_value, i_dir in zip(depth_values, idx_dirs)]
#
#     return depth,(depth[depth>0.5].min(),depth[depth<150].max()),bds_raw,skymask