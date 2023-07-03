import os
import numpy as np
import torch
import torch.multiprocessing as mp

import collections
from tqdm import tqdm, trange

from dataloader.dataloader import load_dataset
from dataloader.rayset import NuscenesDataLoader

from utils.sample_utils import sample_rays
from utils.device_utils import init_devices
from utils.model_utils import learning_rate_decay, build_main_model, build_pose_ref_model, resume_training, exp_dir_prep
from utils.arg_parser import config_parser

from model.models import make_mipnerf
from model.loss_factory import ProposalLoss, RgbLoss, SemanticLoss, SmoothLoss
from model.run_nerf_helpers import *
from model.render import *
from model.loss_factory import *
from model.confidence import Confidence, build_confidence_model, calc_depth_loss


DEBUG = False
np.random.seed(0)
Rays = collections.namedtuple('Rays', ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far','app'))        

def train(rank=None, world_size=None, seed=None):
    parser = config_parser()
    args = parser.parse_args()
    _DEVICE = init_devices(args, rank, world_size)

    #*---------------------=====| Load DataSet |=====-------------------------*# 
    train_depends, bds, render_depends, splits = load_dataset(args)
    images, poses, viewc, intrinsics, depth_gts, flow, cam_index, skymask, seg_masks, semantic = train_depends
    if semantic:
        semantic_index, semantic_labels = semantic 
    
    i_train, i_val, i_test = splits
    N = images.shape[0]
    
    writer = exp_dir_prep(args)

    #*  ---------------------=====| Build Model & Loss Fn |=====------------------------- *# 

    pose_param_net, optimizer_pose, Conf = None, None, None
    model, optimizer, _DEVICE = build_main_model(args, rank, _DEVICE)
    if args.depth_conf:
        Conf, optimizer_conf = build_confidence_model(args, N, _DEVICE)
    if args.pose_refine:
        pose_param_net, optimizer_pose = build_pose_ref_model(args, poses, _DEVICE)
    start, model, optimizer, pose_param_net, optimizer_pose, Conf = resume_training(args, model, optimizer, pose_param_net, optimizer_pose, Conf)

    #* Loss Functions
    rgb_loss_fn = RgbLoss(args)
    semantic_loss_fn = SemanticLoss(args)
    depth_loss_fn = DepthLoss(args)
    smooth_loss_fn = SmoothLoss(args)
    proposal_loss_fn = ProposalLoss(args)

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    
    #*  -------------------------=====| Training |=====------------------------- *# 
    
    start += 1
    psnr = 0.
    depth_loss_log = 0.
    semantic_loss_log = 0.
    semantic_train_time = 0
    if(Conf):
        Conf.train()
    if(pose_param_net):
        pose_param_net.train()
    
    all_conf_maps = None
    if args.depth_conf and args.precompute_conf:
        all_conf_maps = Conf.precompute_conf_map(args, cam_index, images, depth_gts,poses, intrinsics, i_train)
        print('Precompute Confidence Map')
    
    if args.pose_refine:
        args.no_batching=True

    #*  -------------------------=====| Sampling Rays |=====------------------------- *# 
    
    rays_loader = NuscenesDataLoader(args, 
                                     images, 
                                     i_train, 
                                     poses, 
                                     intrinsics, 
                                     depth_gts, 
                                     bds,
                                     cam_index,
                                     batch_n=args.N_rgb, 
                                     device=_DEVICE)
    rays_iter = iter(rays_loader)
    global_step = start

    for i in trange(start, args.N_iters + 1):
        batch_rays, sel_coords, img_i, target_s, target_depth = sample_rays(args, rays_iter, rays_loader, pose_param_net)
        pred = model(batch_rays, 
                     args.randomized,
                     args.white_bkgd,
                     viewc)

        optimizer.zero_grad()
        if args.pose_refine:
            optimizer_pose.zero_grad()
        if args.depth_conf:
            optimizer_conf.zero_grad()
            
        #* Unzip the Pred
        if not args.proposal_loss:
            (rgb_coarse, pred_distance_coarse, _,)= pred[0]
            (rgb_fine, pred_distance, _, semantic_fine)= pred[1]
        else:
            (rgb_coarse, pred_distance_coarse, _, s_vals_c, weights_c)= pred[0]
            (rgb_fine, pred_distance, _, semantic_fine, s_vals_f, weights_f)= pred[1]

        if args.semantic and img_i in semantic_index: 
            target_semantic = semantic_labels[img_i]
            target_semantic = torch.tensor(target_semantic)
            target_semantic = target_semantic[sel_coords[:,0], sel_coords[:,1]]

        # Mask Out Black Area in Waymo left/right cameras
        if args.waymo and (cam_index[img_i]==3 or cam_index[img_i]==4):
            rgb_mask = sel_coords[:,0] < 886
            rgb_fine = rgb_fine[rgb_mask]
            target_s = target_s[rgb_mask]
            
            if args.semantic and img_i in semantic_index:
                semantic_fine = semantic_fine[rgb_mask]
                target_semantic = target_semantic[rgb_mask]
                

    #*  -------------------------=====| Loss Functions |=====------------------------- *# 
        # RGB Loss
        img_loss = rgb_loss_fn(rgb_fine, target_s)
        loss = img_loss

        psnr += mse2psnr(loss)

        sel_coords_smooth = sel_coords[args.N_rgb:,]
        rgb_fine = rgb_fine[:args.N_rgb,]
        sel_coords = sel_coords[:args.N_rgb]
        pred_distance = pred_distance[:args.N_rgb]
        pred_distance_coarse = pred_distance_coarse[:args.N_rgb]
        pred_distance_smooth = pred_distance[args.N_rgb:]

        target_depth = target_depth[:args.N_rgb]
        target_s = target_s[:args.N_rgb,:]

        if args.backcam and cam_index[img_i]==0:
            backcam_mask = sel_coords[:,0]<750
            if args.seg_mask:
                seg_mask = seg_masks[img_i][sel_coords[:,0], sel_coords[:,1]]
                backcam_mask = backcam_mask & (seg_mask==0)
            rgb_fine = rgb_fine[backcam_mask]
            target_s = target_s[backcam_mask]
        
        # Smooth Loss
        if(args.smooth_loss):
            smooth_loss = smooth_loss_fn(images[img_i], skymask[img_i], sel_coords_smooth, pred_distance_smooth)
            loss += smooth_loss
        
        # Proposal Loss
        if args.proposal_loss:
            proposal_loss = proposal_loss_fn(s_vals_f, weights_f, s_vals_c, weights_c)
            loss +=  proposal_loss

        # Semantic Loss
        if args.semantic and img_i in semantic_index:
            semantic_fine = semantic_fine[:args.N_rgb]
            target_semantic = target_semantic[:args.N_rgb]
            semantic_loss = semantic_loss_fn(semantic_fine, target_semantic.long().squeeze(-1))
            semantic_loss_log += semantic_loss
            semantic_train_time += 1
            loss += semantic_loss

        # Depth Loss
        if(args.depth_loss):
            confidence_depends = [Conf,                     # Confidence Model
                                  sel_coords,               # Target Coordinations
                                  img_i,                    # Image Index
                                  cam_index,                # Corresponding Camera Index
                                  images, depth_gts,        # Image & Depth GT
                                  poses, intrinsics,        # Camera Extrinsic & Intrinsic
                                  i_train,                  # Training Split
                                  pose_param_net,           # Pose refine Network
                                  all_conf_maps, skymask    # Utils
                                ]
            depth_loss = calc_depth_loss(args, 
                                         backcam_mask, 
                                         target_depth, pred_distance, pred_distance_coarse, 
                                         depth_loss_fn, confidence_depends)
            loss += depth_loss * args.depth_lambda
            depth_loss_log += depth_loss
        
        try:
            loss.backward()
        except RuntimeError:
            import pdb; pdb.set_trace()

        optimizer.step()
        if args.pose_refine:
            optimizer_pose.step()
        if args.depth_conf:
            optimizer_conf.step()

        #*  ---------------------=====| Logger & Checkpoint |=====------------------------- *# 

        writer.add_scalar('RGB Loss', img_loss, i)
        writer.add_scalar('PSNR', mse2psnr(img_loss), i)
        if args.depth_conf:
            for k in range(Conf.lambdas.shape[0]):
                writer.add_histogram('conf_{:01d}'.format(k),Conf.lambdas[k],i)
        
        if args.pose_refine:
            rotation=pose_param_net.state_dict()['r'].cpu().detach().numpy()
            writer.add_histogram(tag='rotation_x', values=rotation[:,0], global_step=i)
            writer.add_histogram(tag='rotation_y', values=rotation[:,1], global_step=i)
            writer.add_histogram(tag='rotation_z', values=rotation[:,2], global_step=i)
   

        if args.depth_loss:
            writer.add_scalar('Depth Loss', depth_loss_log/args.i_print)
        if args.semantic and semantic_train_time != 0:
            writer.add_scalar('Semantic Loss', semantic_loss_log/semantic_train_time)
        
        new_lrate = learning_rate_decay(i, 
                        lr_init=5e-4, lr_final=5e-6, 
                        max_steps=200000,
                        lr_delay_steps=2500, lr_delay_mult=0.01)
        
        optimizer.param_groups[0]['lr'] = new_lrate

        if not args.distributed or rank == 0:
            if i % args.i_print == 0 and i>0:
                print('args.expname:',args.expname)
                print(f"[Step:{i}]: Total_loss:{loss.item()}", end=" | ")
                if args.depth_loss:
                    print(f"Depth_loss:{depth_loss_log/args.i_print}", end=" | ")
                    depth_loss_log = 0
                if args.semantic:
                    print(f"Semantic_loss:{semantic_loss_log/semantic_train_time}", end=" | ")
                    semantic_loss_log = 0

                print(f"PSNR:{psnr/args.i_print}")
                psnr=0
                # print(int(os.environ["RANK"]))
            if i % args.i_weights==0 and i!=0:
                torch.save(
                    {
                        'global_step':i,
                        'model_param':model.state_dict(),
                        'optimzer':optimizer.state_dict(),
                        'confidence': Conf.state_dict() if args.depth_conf else None,
                        'optimizer_conf': optimizer_conf.state_dict() if args.depth_conf else None,
                    },
                    os.path.join(args.basedir, args.expname, '{:06d}.tar'.format(i)))
                
                if args.pose_refine:
                    torch.save( {
                        'model_param':pose_param_net.state_dict(),
                        'optimizer':optimizer_pose.state_dict()
                    },
                    os.path.join(args.basedir, args.expname, 'pose','{:06d}.tar'.format(i)))
                
        global_step += 1

def train_from_folder_distributed():
    world_size = torch.cuda.device_count()
    print(f"world_size: {world_size}")
    global_seed = 0
    distributed = torch.cuda.device_count() >1
    if distributed:
        mp.spawn(train,
                args=(world_size, global_seed),
                nprocs=torch.cuda.device_count(),
                join=True)
    else:
        print('training alone')
        train(torch.cuda.current_device(),world_size, global_seed)

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
