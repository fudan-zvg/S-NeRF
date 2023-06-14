import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import imageio
import cv2
import matplotlib.pyplot as plt

from utils.vis_tools import visualize_gray
from utils.pytorch_msssim import msssim, ssim
from torchvision import models
from dataloader.load_nuscenes import recenter_poses
from model.confidence import calc_final_confidence

def edge_aware_loss_v2(rgb, disp,skymask=None):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    mean_disp = disp.mean(1, True).mean(2, True)
    disp = disp / (mean_disp + 1e-7)

    grad_disp_x = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
    grad_disp_y = torch.abs(disp[:, :-1, :, :] - disp[:, 1:, :, :])

    grad_rgb_x = torch.mean(torch.abs(rgb[:, :, :-1, :] - rgb[:, :, 1:, :]), 3, keepdim=True)
    grad_rgb_y = torch.mean(torch.abs(rgb[:, :-1, :, :] - rgb[:, 1:, :, :]), 3, keepdim=True)

    grad_disp_x *= torch.exp(-grad_rgb_x)
    grad_disp_y *= torch.exp(-grad_rgb_y)
    # mask=torch.ones_like(disp)

    if skymask is not None:
        grad_disp_x+=skymask[:,:,:-1,:]*grad_disp_x
        grad_disp_y+=skymask[:,:-1,:,:]*grad_disp_y
    return grad_disp_x.mean() + grad_disp_y.mean()

def vis_proj_points(base_img, tgt_img, base_coord, tgt_coord):

    b_i = base_img.cpu().detach().numpy()*255
    t_i = tgt_img.cpu().detach().numpy()*255
    for pt in base_coord.tolist():
        b_i = cv2.circle(b_i, pt, 2, (255,0,0), thickness=-1)
    for pt in tgt_coord.round().int().tolist():
        t_i = cv2.circle(t_i, pt, 2, (255,0,0), thickness=-1)
    imageio.imwrite('./_base.png', b_i)
    imageio.imwrite('./_tgt.png', t_i)

def vis_conf_map(H, W, conf_map, coord, mode):
    canvas = torch.zeros((H, W))
    canvas[coord[:,1], coord[:,0]] = conf_map
    canvas = canvas.cpu().detach().numpy()
    mask = canvas==0
    conf_color = visualize_gray(canvas)
    conf_color[mask] = 0
    imageio.imwrite(f'./vis/conf_maps/{mode}_conf_map.png', conf_color)

def vis_reprojection_err(data_path, base_idx, tgt_idx, type, pose_refine_network=None):
    '''
    Visualize the reprojection err between two images
    Parameters:
        data_path: [str] Data PATH
        base_idx: [int] base data idx
        tgt_idx: [int] base data idx
        type: ['ori'/'refine'] use original poses or refined poses
    '''
    assert type in ['ori', 'refine']
    poses_arr=np.load(f'{data_path}/poses_bounds.npy')
    depthes=[cv2.imread(f'{data_path}/depth/000{i:02}.png',-1)/256. for i in range(poses_arr.shape[0])]
    
    hwf=poses_arr[3,:-4].reshape(3,5)[:3,4]
    import pdb; pdb.set_trace()
    # TODO multiple intrinsics
    
    factor=900./288.
    cx, cy, focal = hwf[0]*factor, hwf[1]*factor, hwf[2]*factor

    intr=np.eye(3);
    intr[0,0], intr[1,1] = focal, focal
    intr[0,2], intr[1,2] = cx, cy

    poses = poses_arr[:, :-4].reshape([-1, 3, 5])
    poses = np.concatenate(
            [poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)
    poses, c2w = recenter_poses(poses)
    
    base_depth = depthes[base_idx]
    base_coord=torch.tensor(base_depth).nonzero()
    base_coord = torch.flip(base_coord, [1])

    imgs = np.stack([base_depth,depthes[tgt_idx]],axis=0)
    
    if not pose_refine_network:
        poses = np.stack([poses[base_idx], poses[tgt_idx]])  
    else:
        pose_refine_network.eval()
        b_p = pose_refine_network(base_idx)
        t_p = pose_refine_network(tgt_idx)
        poses = np.stack([b_p, t_p])

    imgs = torch.tensor(imgs)
    poses = torch.tensor(poses)
    intrs = torch.stack([torch.tensor(intr)]*2)

    _, ori_img, tgt_img, _ = reproj_err( base_depth[base_depth!=0], 
                                base_coord, 
                                imgs[...,None], 
                                poses.float(), 
                                intrs,
                                ret_full=True,
                                modes=['depth'])

    err = (ori_img-tgt_img).abs()
    err[ori_img!=0] /= ori_img[ori_img!=0]
    
    plt.figure(figsize=(16, 9))
    res = seaborn.heatmap(err[:,:,0]>0.05)
    imageio.imwrite('./ori_depth.png', ori_rgb_img.cpu().numpy())
    imageio.imwrite('./tgt_rgb.png', tgt_rgb_img.cpu().numpy())
    fig = res.get_figure()
    fig.savefig(f'./error_map_{base_idx}-{tgt_idx}.png')
    import pdb; pdb.set_trace()

def sample_points(tgt_img, tgt_coord, align_corners=False):
    # Sample Point
    H, W = tgt_img.shape[:2]
    tgt_img = tgt_img.unsqueeze(0).float().permute(0,3,1,2)

    grid = tgt_coord.clone()
    # scale to [-1,1]
    grid[:, 1] = (grid[:, 1] / ((H-1)//2)) - 1
    grid[:, 0] = (grid[:, 0] / ((W-1)//2)) - 1
    
    grid = grid[None, ...].unsqueeze(2).float() # [N, P, 1, 2]
    tgt_rgb = F.grid_sample(tgt_img, grid, align_corners=align_corners)
    tgt_rgb = tgt_rgb.squeeze(3).squeeze(0)
    return tgt_rgb.T


def warping(proj_coord, depends):
    imgs, depths, poses, intrs = depends
    base_img, tgt_img = imgs
    base_depth, tgt_depth = depths
    H, W = base_img.shape[:2]
    base_pose, tgt_pose = poses
    base_intr, tgt_intr = intrs
    
    focal = torch.tensor([base_intr[0][0], base_intr[1][1]]).mean()
    tgt_focal = torch.tensor([tgt_intr[0][0], tgt_intr[1][1]]).mean()
    i, j = (proj_coord[:,0] - base_intr[0,2])/focal,  -(proj_coord[:,1] - base_intr[1,2])/focal

    dirs = torch.stack([i,j,-torch.ones_like(i)],-1).permute(1,0) * base_depth.reshape(-1,1)[:,0]
    dirs = F.pad(dirs.permute(1,0), (0,1), 'constant', 1).float()

    pts = base_pose @ dirs.T
    tgt_cam_pt = (torch.inverse(tgt_pose) @ pts)[:3].T

    dep = tgt_cam_pt[:, 2].clone().abs().detach()
    tgt_cam_pt[:, 0] = tgt_cam_pt[:, 0] / dep
    tgt_cam_pt[:, 1] = tgt_cam_pt[:, 1] / dep
    tgt_coord = tgt_cam_pt[:, :2] * tgt_focal
    tgt_coord[:,0] = tgt_coord[:,0] + tgt_intr[0,2]
    tgt_coord[:,1] = -tgt_coord[:,1] + tgt_intr[1,2]

    x, y = tgt_coord[:,1].round().long(), tgt_coord[:,0].round().long()
    
    # Mask Overflow Pixels
    x_mask= (x>=0) & (x<H)
    y_mask= (y>=0) & (y<W)
    mask=x_mask & y_mask
    x, y = x[mask], y[mask]

    sample_rgb = sample_points(tgt_img, tgt_coord[mask])
    tgt_depth = tgt_depth[x, y]
    fake_depth = dep[mask]

    fake_img = torch.zeros_like(base_img).to(base_img.device)
    coord = proj_coord[mask]
    fake_img[coord[:,1], coord[:,0]] = sample_rgb

    return fake_img, tgt_depth, fake_depth, mask.reshape(H, W)

def reproj_flow_err(proj_coord, depends,loss_type='l2'):
    if loss_type == 'l1':
        loss_fn = lambda x, y : (x-y).abs()
    elif loss_type == 'l2':
        loss_fn = lambda x, y : ((x-y)**2)
    else:
        raise NotImplementedError
    imgs, depths, poses, intrs,reproj_flow = depends
    base_img, tgt_img = imgs
    base_depth, tgt_depth = depths
   
        
    H, W = base_img.shape[:2]
    base_pose, tgt_pose = poses
    base_intr, tgt_intr = intrs
    
    focal = torch.tensor([base_intr[0][0], base_intr[1][1]]).mean()
    tgt_focal = torch.tensor([tgt_intr[0][0], tgt_intr[1][1]]).mean()
    i, j = (proj_coord[:,0] - base_intr[0,2])/focal,  -(proj_coord[:,1] - base_intr[1,2])/focal

    dirs = torch.stack([i,j,-torch.ones_like(i)],-1).permute(1,0) * base_depth.reshape(-1,1)[:,0]
    dirs = F.pad(dirs.permute(1,0), (0,1), 'constant', 1).float()

    pts = base_pose @ dirs.T
    tgt_cam_pt = (torch.inverse(tgt_pose) @ pts)[:3].T

    dep = tgt_cam_pt[:, 2].clone().abs().detach()
    tgt_cam_pt[:, 0] = tgt_cam_pt[:, 0] / dep
    tgt_cam_pt[:, 1] = tgt_cam_pt[:, 1] / dep
    tgt_coord = tgt_cam_pt[:, :2] * tgt_focal
    tgt_coord[:,0] = tgt_coord[:,0] + tgt_intr[0,2]
    tgt_coord[:,1] = -tgt_coord[:,1] + tgt_intr[1,2]

    x, y = tgt_coord[:,1].round().long(), tgt_coord[:,0].round().long()
    import pdb;pdb.set_trace()
    return None

def reproj_err(modes, proj_coord, depends, ret_full=False, loss_type='l1', vgg_loss=None):
    '''
    Calc the reprojection error between multiple frames
    '''
    imgs, depths, poses, intrs = depends
    base_img, tgt_img = imgs
    base_depth, tgt_depth = depths
    H, W = base_img.shape[:2]
    assert loss_type in ['l1', 'l2']
    
    if loss_type == 'l1':
        loss_fn = lambda x, y : (x-y).abs()
    elif loss_type == 'l2':
        loss_fn = lambda x, y : ((x-y)**2)
    else:
        raise NotImplementedError

    error_dict = {}
    
    fake_img, tgt_depth, fake_depth, mask = warping(proj_coord, depends)

    base_img_ = base_img.clone()
    base_depth_ = base_depth.clone()
    base_img_[~mask] = 0
    base_depth_[~mask] = 0
   
    if 'rgb' in modes:
        rgb_err_map = loss_fn(base_img_, fake_img).mean(2)
        # vis_conf_map(H, W, rgb_err_map.reshape(1,-1)[0], proj_coord, 'rgb_err')
        error_dict['rgb'] = rgb_err_map[mask].reshape(1,-1)[0]

    if 'ssim' in modes:
        ssim_map = ssim(base_img_[None, ...].permute(0,3,1,2), fake_img[None, ...].permute(0,3,1,2), full=True)
        ssim_err_map = (1-ssim_map.mean(1))[0]
        # vis_conf_map(H, W, ssim_err_map.reshape(1,-1)[0], proj_coord, 'ssim_err')
        error_dict['ssim'] = ssim_err_map[mask].reshape(1,-1)[0]
    
    if 'depth' in modes:
        depth_err_map = loss_fn(fake_depth, tgt_depth)/torch.max(tgt_depth, torch.tensor(1e-10))
        error_dict['depth'] = depth_err_map
        
    if 'vgg' in modes and vgg_loss:
        with torch.no_grad():
            vgg_err_map = vgg_loss(base_img_, fake_img)
        # vis_conf_map(H, W, vgg_err_map.reshape(1,-1)[0], proj_coord, 'vgg_err')
        error_dict['vgg'] = vgg_err_map[mask].reshape(1,-1)[0]
        
    if(ret_full):
        return error_dict, base_img, fake_img, mask

    return error_dict, mask


def get_reproj_conf(modes, proj_coord, base_depends, repj_depends,flow_depends=None, vgg_loss=None, tau=0.2,loss_type='l1'):
    
    base_img, base_depth, base_pose, base_intr = base_depends
    repj_imgs, repj_depths, repj_poses, repj_intrs = repj_depends
    if flow_depends is not None:
        flow_imgs,flow_depths,flow_poses,flow_intrs,repj_flows = flow_depends
    masks = torch.zeros(proj_coord.shape[0]).to(base_img.device)
    H, W = base_img.shape[:2]   

    confs, masks = {}, {}
    for mode in modes:
        confs[mode] = torch.zeros(proj_coord.shape[0]).to(base_img.device)
        masks[mode] = torch.zeros(proj_coord.shape[0]).to(base_img.device)
    
    fake_img, fake_depth = torch.zeros_like(base_img), torch.zeros_like(base_depth)
    depth_mask=None
    for repj_img, repj_dep, repj_pose, repj_intr in zip(repj_imgs, repj_depths, repj_poses, repj_intrs):
        imgs = torch.stack((base_img, repj_img))
        depths = torch.stack((base_depth, repj_dep))
        poses = torch.stack((base_pose, repj_pose))
        intrs = torch.stack((base_intr, repj_intr))
        depends = [imgs, depths, poses, intrs]
        err_dict, mask = reproj_err(modes, proj_coord, depends, ret_full=False, loss_type='l1', vgg_loss=vgg_loss)
        Mask = mask.reshape(1,-1)[0]
        for mode in modes:
            if mode in err_dict.keys():
                if mode=='depth':
                    depth_mask=err_dict['depth']>tau
                    err_dict['depth']=torch.clamp(err_dict['depth'],max=tau)
                
                conf = err_dict[mode].max() - err_dict[mode]
                conf = conf / conf.max()
                confs[mode][Mask] += conf
                masks[mode][Mask] += 1
                
    if flow_depends is not None:
        for flow_img,flow_depth,flow_pose,flow_intr,repj_flow in zip(flow_imgs,flow_depths,flow_poses,flow_intrs,repj_flows):
            #TODO complete flow
            imgs = torch.stack((base_img, flow_img))
            depths = torch.stack((base_depth, flow_depth))
            poses = torch.stack((base_pose, flow_pose))
            intrs = torch.stack((base_intr, flow_intr))
            depends = [imgs, depths, poses, intrs,repj_flow]
            flow_err,mask=reproj_flow_err(proj_coord,depends,loss_type='l2')
            
    masks[masks==0] = 1
    
    for mode in modes:
        mask = masks[mode]
        conf = confs[mode]
        mask[mask==0]=1
        conf /= mask
        confs[mode]=conf
        if depth_mask is not None:
            temp=confs[mode][Mask]
            temp[depth_mask]=0.
            confs[mode][Mask]=temp
    return confs


def calc_depth_loss(args, backcam_mask, target_depth, pred_distance, pred_distance_coarse, depth_loss_fn, confidence_depends):
    img_i, cam_index = confidence_depends[2], confidence_depends[3]
    target_mask = target_depth!=0
    target_depth = target_depth.unsqueeze(-1)
    pred_distance = pred_distance.unsqueeze(-1)
    pred_distance_coarse = pred_distance_coarse.unsqueeze(-1)

    depth_loss = depth_loss_fn(pred_distance[target_mask], pred_distance_coarse[target_mask], target_depth[target_mask])

    if(args.depth_conf): # this time is the single image training mode
        confidence = calc_final_confidence(args, target_mask, confidence_depends):
        depth_loss *= confidence
    
    if args.backcam and cam_index[img_i] == 0:
        depth_loss = depth_loss[backcam_mask[target_mask]]
    return depth_loss.mean()