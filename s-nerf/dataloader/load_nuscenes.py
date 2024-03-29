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
    poses = np.concatenate([P[:, :, 1:2],P[:, :, 0:1], -P[:, :, 2:3], P[:, :, 3:4]], 2) # convert [r, -u, t] to [-u, r, -t]
    poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)  # convert [-u, r, -t] to [r, u, -t]
    return poses

def load_nuscenes_data(args, bds_raw, bd_factor=.75):
    render_focal = None
    render_depth = None
    path=args.datadir
    sc = 1. if bd_factor==0. else 1./(bds_raw.min() * bd_factor)

    imgdir = os.path.join(path, 'images')
    img_files = sorted(os.listdir(imgdir), key=lambda x : int(x.split('.')[0]))
    imgfiles = [os.path.join(imgdir, f) for f in img_files if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    imgs = [imageio.imread(f)[...,:3].astype(np.float32)/255. for f in imgfiles]
    imgs = np.stack(imgs, 0)
    num = imgs.shape[0]

    with open(os.path.join(args.datadir, 'poses_bounds.npy'),
                                'rb') as fp:
            poses_arr = np.load(fp).astype(np.float32)
    
    print('poses_arr',poses_arr.shape)
    print('Colmap:',args.colmap)
    ## colmap:n*17 nuscenes:n*19
    if args.colmap:
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        bds = poses_arr[:, -2:].transpose([1, 0])
        raw_hw=poses[:,:2,4]
    else:
        poses = poses_arr[:, :-4].reshape([-1, 3, 5])
        bds = poses_arr[:, -4:-2].transpose([1, 0])
        raw_hw=poses_arr[:, -2:].transpose([1, 0])
    if poses.shape[0] != imgs.shape[0]:
        raise RuntimeError('Mismatch between imgs {} and poses {}'.format(
            imgs.shape[-1], poses.shape[-1]))
    raw_cam_K=poses[:,:,4].copy().astype(np.float32).transpose([1,0]);
    factor=raw_hw[0,0]/imgs.shape[1]
    print('factor',factor)
    if args.colmap:
        cx=raw_cam_K[1,:]/factor*.5
        cy=raw_cam_K[0,:]/factor*.5
        focal=raw_cam_K[2,:]/factor
    else:
        cx=raw_cam_K[0,:]/factor
        cy=raw_cam_K[1,:]/factor
        focal=raw_cam_K[2,:]/factor
    K = [np.array([[focal[i],0,cx[i]],[0,focal[i],cy[i]],[0,0,1]]) for i in range(num)]
    K = np.stack(K, 0)
    poses = np.concatenate(
            [poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)
    
    poses[:, :3, 3] *= sc
    bds = bds_raw * sc
    poses, c2w = recenter_poses(poses)
    if not args.no_align:
        poses[:,:3,3]-=poses[0,:3,3]

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
    semantic = np.load(semantic_path)
    semantic_labels = semantic[...,1:]
    semantic_index = semantic[:,0,0,0,0]
    return semantic_index, semantic_labels
    
def load_depth_map(path, H, W, bd_factor=.75, sky_mask=False):
    skymask = None
    depth_dir = os.path.join(path, 'depths')
    imgdir = depth_dir
    img_files = sorted(os.listdir(imgdir), key=lambda x : int(x.split('.')[0]))
    imgfiles = [os.path.join(imgdir, f) for f in img_files if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    imgs = [cv2.resize(cv2.imread(f,-1)/256.,(W,H)) for f in imgfiles]
    depth = np.stack(imgs, 0).astype(np.float32)
    
    min_thresh = 0.5
    max_thresh = 200
    if sky_mask:
        skymask = depth>max_thresh
    
    depth[depth>min_thresh] = np.clip(depth[depth>min_thresh], 
                                  a_min=max(depth[depth>min_thresh].min(),2), 
                                  a_max=100)
    
    bds_raw = np.array([[max(dep[dep>min_thresh].min(),2.), dep[dep<max_thresh].max()] for dep in depth])
    sc = 1. if bd_factor == 0. else 1./(bds_raw.min() * bd_factor)
    depth = depth * sc
    bds = (depth[depth>min_thresh].min(), depth[depth<max_thresh].max())
    
    return depth, bds, bds_raw, skymask