
import torch
import numpy as np
import os
from model.models import make_mipnerf
import model.math_ops as math_ops
from torch import nn
import torch.multiprocessing as mp
import utils.render_utils as utils
import model.models as models
import functools
from os import path
from utils.arg_parser import config_parser
from dataloader.dataloader import load_dataset
from utils.sample_utils import get_rays_single_img,get_rays_single_img_
from model.poses import LearnPose
# from vis import visualize_depth,visualize_semantic
from tqdm import tqdm, trange
from utils.sample_utils import sample_single_img
from model.run_nerf_helpers import *
import cv2
import collections

Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))

def pred2real(pred_distance,near,far):
    real_ditance=1./(pred_distance[...,None]/far+(1.-pred_distance)[...,None]/near)
    return  real_ditance

def eval(rank=None):
    parser = config_parser()
    args = parser.parse_args()
    args.proposal_loss=False
    basedir = args.basedir
    expname = args.expname
    ckpt_num=args.ckpt
    testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(ckpt_num))
    os.makedirs(testsavedir, exist_ok=True)
    # load model from ckpt
    
    train_depends, bds, render_depends, splits = load_dataset(args)
    if args.dataset_type == 'llff':
        images, hwf, poses, ori_points, depth_gts = train_depends
        intrinsics = None
    else:
        images, poses, intrinsics, depth_gts, flow, cam_index, skymask,semantic = train_depends

    viewc = poses[:,:3,3].mean(axis=0)
    near,far = depth_gts[depth_gts>0.5].min(),depth_gts.max() ## the sky is at distance=100
    print('viewc is:',viewc)
    render_poses, render_depth, render_focal = render_depends
    _DEVICE=torch.cuda.current_device()
    i_train, i_val, i_test = splits
    if args.half_test:
        i_test=i_test[::2]

    poses = torch.tensor(poses).to(_DEVICE)
    near = torch.tensor(near).to(_DEVICE)
    far = torch.tensor(far).to(_DEVICE)
    if intrinsics is not None:
        intrinsics = torch.tensor(intrinsics).to(_DEVICE)

    viewc = torch.tensor(viewc).to(_DEVICE)
    with torch.no_grad():
        model = make_mipnerf(
            args,
            device=rank)
        checkpoint=torch.load(os.path.join(basedir,expname,'{:06d}.tar'.format(ckpt_num)))
        
        model =nn.DataParallel(model, device_ids=[0], output_device=rank,
                    )
        model.load_state_dict(checkpoint['model_param'])
        render_fn=functools.partial(model,randomized=args.randomized,
            white_bg=args.white_bkgd,viewc=viewc)
    ## use test refine finetuning.
    log_loss=0.
    if args.test_refine_iter:
        #* Fixed the NeRF MLP Parameters
        model.require_grad = False
        model.eval()
        pose_param_net = LearnPose(num_cams=poses.shape[0], learn_R=True, learn_t=False, init_c2w=poses)
        pose_param_net = pose_param_net.to(_DEVICE)
        optimizer_pose = torch.optim.Adam(pose_param_net.parameters(), lr=1e-4)
        for i in trange(0, args.test_refine_iter):
            img_i = np.random.choice(i_test)
            pose = pose_param_net(img_i)
            batch_rays, target_s, _,sel_coords,_=sample_single_img(args, torch.tensor(images[img_i]).to(_DEVICE), torch.tensor(depth_gts[img_i]).to(_DEVICE), pose, intrinsics[img_i],near,far)
            pred = model(batch_rays, args.randomized,
                    args.white_bkgd,viewc)
            (rgb_coarse, _, _)= pred[0]
            (rgb_fine, pred_distance, _,_)= pred[1]
            
            #* Only Test
            optimizer_pose.zero_grad()
            if args.waymo and (cam_index[img_i]==3 or cam_index[img_i]==4):
                rgb_mask= sel_coords[:,0] < 886
                rgb_fine=rgb_fine[rgb_mask];target_s=target_s[rgb_mask] 
            img_loss = img2mse(rgb_fine, target_s)
            loss = img_loss
            log_loss += loss
            loss.backward()
            optimizer_pose.step()

            if(i%10 == 0) and (i >0):
                tqdm.write(f"[Pose Finetune] Iter: [{i}/{args.test_refine_iter}], Loss: {log_loss.item()/10}, PSNR: {mse2psnr(img2mse(rgb_fine, target_s))}")
                log_loss = .0

        new_poses = []
        for idx in i_test:
            pose = pose_param_net(idx)
            new_poses.append(pose)
        new_poses = torch.stack(new_poses)  
    

    with torch.no_grad():
        if args.pose_refine:
            if not args.translation:
                pose_param_net = LearnPose(num_cams=poses.shape[0], learn_R=True, learn_t=False, init_c2w=poses)
            else:
                pose_param_net = LearnPose(num_cams=poses.shape[0], learn_R=True, learn_t=True, init_c2w=poses)
            pose_param_net=pose_param_net.to(_DEVICE)
            pose_ckpt = torch.load(os.path.join(basedir,expname,'pose','{:06d}.tar'.format(ckpt_num)))
            pose_param_net.load_state_dict(pose_ckpt['model_param'])
            
        ##################################### render testset
        if args.eval_test:
            PSNR=[]
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(ckpt_num))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            for idx in range(len(i_test)):
                # idx=6
                index=i_test[idx]
                
                print(f'Evaluating {idx+1}/{len(i_test)}')
                if args.test_refine_iter:
                    batch_rays=get_rays_single_img(args, torch.tensor(images[index]).to(_DEVICE),torch.tensor(depth_gts[index]).to(_DEVICE), new_poses[idx], intrinsics[index],near=near,far=far, factor=args.render_factor)
                else:
                    if depth_gts is not None:
                        batch_rays=get_rays_single_img(args,torch.tensor(images[index]).to(_DEVICE), torch.tensor(depth_gts[index]).to(_DEVICE), poses[index], intrinsics[index],near=near,far=far, factor=args.render_factor)
                    else:
                        batch_rays= get_rays_single_img_(args, images[index], poses[index], hwf, bds)

                pred_color, pred_distance, pred_acc,pred_semantic = models.render_image(render_fn, batch_rays, rank, chunk=4096)
                batch_pixels=images[index]
                
                # TODO Fix Hard Code
                H,W = images.shape[1:3]
                batch_pixels = torch.tensor(cv2.resize(batch_pixels, (W//args.render_factor,H//args.render_factor)))
                psnr = float(
                    math_ops.mse_to_psnr(((pred_color - batch_pixels)**2).mean()))
                print('psnr:',psnr)
                PSNR.append(psnr)
                # utils.save_img_uint8(pred_color.cpu().numpy(), path.join(testsavedir, 'ablation_full.png'))
                utils.save_img_uint8(pred_color.cpu().numpy(), path.join(testsavedir, 'color_{:03d}.png'.format(idx)))
                if not args.real:
                    real_distance=pred2real(pred_distance,batch_rays.near,batch_rays.far)
                else:
                    real_distance=pred_distance
                Depth=real_distance.cpu().numpy()
                np.save(path.join(testsavedir,'depth_{:03d}.npy').format(idx),Depth[:,:])
                # visualize_depth(Depth[:,:],os.path.join(expname,'testset_{:06d}').format(ckpt_num),idx)
                if args.semantic:
                    logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
                    Semantic = logits_2_label(pred_semantic).cpu().numpy()
                    np.save(path.join(testsavedir,'semantic_{:03d}.npy').format(idx),Semantic[:,:])                    
                    # visualize_semantic(Semantic[:,:],os.path.join(expname,'testset_{:06d}').format(ckpt_num),idx)
            print('psnr',np.array(PSNR).mean())

        ################################### render trainset
        if args.eval_train:
            PSNR=[]
            trainsavedir = os.path.join(basedir, expname, 'trainset_{:06d}'.format(ckpt_num))
            os.makedirs(trainsavedir, exist_ok=True)
            print('train poses shape', poses[i_train].shape)
            for idx in range(len(i_train)):
                print(f'Evaluating {idx+1}/{len(i_train)}')
                index=i_train[idx]
                pose = pose_param_net(index) if args.pose_refine else poses[index, :3,:4]
                if depth_gts is not None:
                    batch_rays=get_rays_single_img(args, images[index], depth_gts[index], pose, intrinsics[index],near=near,far=far, factor=args.render_factor)
                else:
                    batch_rays= get_rays_single_img_(args, images[index], pose, hwf, bds, factor=args.render_factor)

                pred_color, pred_distance, pred_acc = models.render_image(render_fn, batch_rays, rank, chunk=2048)
                batch_pixels=images[index]
                batch_pixels = torch.tensor(cv2.resize(batch_pixels.cpu().numpy(), (1600//args.render_factor,900//args.render_factor)))
                psnr = float(
                    math_ops.mse_to_psnr(((pred_color - batch_pixels)**2).mean()))
                print('psnr:',psnr)
                PSNR.append(psnr)
                utils.save_img_uint8(
                    pred_color.cpu().numpy(), path.join(trainsavedir, 'color_{:03d}.png'.format(idx)))

                if not args.real:
                    real_distance=pred2real(pred_distance,batch_rays.near,batch_rays.far)
                else:
                    real_distance=pred_distance
                Depth=real_distance.cpu().numpy()
                np.save(path.join(trainsavedir,'depth_{:03d}.npy').format(idx),Depth[:,:])
                visualize_depth(Depth[:,:],os.path.join(expname,'trainset_{:06d}').format(ckpt_num),idx)

            print('psnr',np.array(PSNR).mean())
    

def train_from_folder_distributed():
    world_size = torch.cuda.device_count()
    print(f"world_size: {world_size}")
    global_seed = 0
    initialize(config_path="configs_hydra", job_name="test_app")
    cfg = compose(config_name="eval")
    eval(torch.cuda.current_device(), world_size, global_seed)


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    eval()