import os, sys
import numpy as np
import imageio
import torch
from tqdm import tqdm, trange
import time

from dataloader.dataloader import load_dataset
from torch.utils.tensorboard import SummaryWriter

from model.render_utils import render_only
from model.run_nerf_helpers import *
from model.render import *
from model.loss import edge_aware_loss_v2
from model.confidence import Confidence, select_conf_depends
from model.poses import LearnPose
from utils.sample_utils import sample_rays, sample_single_img

from dataloader.rayset import get_next_batch, init_dataloader, PatchRayDataset, RayDataset, ImageRayDataset

from utils.arg_parser import config_parser
from utils.test_utils import calc_bbox_benchmark, calc_mask_psnr


np.random.seed(0)
DEBUG = False
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    ori_points = None
    parser = config_parser()
    args = parser.parse_args()
    if(args.pose_refine):
        args.no_batching = True
    if(args.smooth_loss):
        args.N_depth = args.patch_sz ** 2 * args.N_patch
    print(f"N_rgb: {args.N_rgb}, N_depth: {args.N_depth}")
    print(f"block_bg: {args.block_bg}, pose_refine: {args.pose_refine}, depth_conf: {args.depth_conf}, smooth_loss: {args.smooth_loss}")

    #*---------------------=====| Load DataSet |=====-------------------------*# 
    train_depends, bds, render_depends, splits, test_mask = load_dataset(args)

    images, poses, intrinsics, depth_gts, depth_ori, bboxes, _ = train_depends
    if(render_depends):
        render_poses, render_depth, render_focal, render_dist = render_depends
        
    i_train, i_val, i_test = splits
    H, W = images[0].shape[:2]

    if args.render_test:
        render_poses = np.array(poses[i_test])
        render_depends = [render_poses, render_depth, render_focal, None]
    elif args.render_train:
        render_poses = np.array(poses[i_train])
        render_depends = [render_poses, render_depth, render_focal, None]

    #*  ---------------------=====| Prep Exp Dir |=====------------------------- *# 
    
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    writer = SummaryWriter(os.path.join(basedir, expname))

    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())


    #*  ---------------------=====| Build Model |=====------------------------- *# 

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, ckpt = create_nerf(args)
    bds_dict = {
        'near' : bds.min() * 0.95,
        'far' : bds.max() * 1.05,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    pose_param_net = None
    if(args.pose_refine):
        pose_param_net = LearnPose(num_cams=poses.shape[0], learn_R=True, learn_t=True, init_c2w=poses)
        pose_param_net = pose_param_net.to(_DEVICE)
        optimizer_pose = torch.optim.Adam(pose_param_net.parameters(), lr=1e-4)
    Conf = Confidence(['rgb', 'ssim'],
                        device=_DEVICE, images_num=images.shape[0])
    optimizer_conf=torch.optim.Adam([
        {'params':[p for p in Conf.parameters() if p.requires_grad], 'lr':1e-4}],
    )
    global_step = start

    # Short circuit if only rendering out from trained model
    if args.render_only or args.render_train:
        print('RENDER ONLY')
        with torch.no_grad():
            render_only(args, render_kwargs_test, basedir, expname, render_depends, train_depends, pose_param_net, splits, start)
            return

    #*  -------------------------=====| Sampling Rays |=====------------------------- *# 
    # Presample If Batching / w/o pose refine
    if not args.no_batching:
        rays_rgbs, rays_depths = sample_rays(args, images, i_train, poses, intrinsics, depth_gts, bboxes)

        if args.random_sample:
            raysRGB_iter = init_dataloader(rays_rgbs, RayDataset, batch_n=args.N_rgb, device=_DEVICE)
            if(args.depth_loss):
                raysDepth_iter = init_dataloader(rays_depths, RayDataset, batch_n=args.N_depth, device=_DEVICE)
        else:
            raysRGB_iter = init_dataloader(rays_rgbs, PatchRayDataset, N_patch=args.N_patch,  patch_sz=args.patch_sz ,device=_DEVICE)
            if(args.depth_loss):
                raysDepth_iter = init_dataloader(rays_depths, ImageRayDataset, batch_n=args.N_depth, device=_DEVICE)

    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    
    #*  -------------------------=====| Training |=====------------------------- *# 
    start = start + 1
    psnr = 0.
    if(args.pose_refine):
        pose_param_net.train()
    if(args.smooth_loss):
        depth_ori = torch.tensor(depth_ori).to(_DEVICE)

    images = torch.tensor(images).to(_DEVICE)
    poses = torch.tensor(poses).to(_DEVICE)
    intrinsics = torch.tensor(intrinsics).to(_DEVICE)
    for dep_gt in depth_gts:
        dep_gt['coord'] = torch.tensor(dep_gt['coord'].copy()).to(_DEVICE)
        dep_gt['depth'] = torch.tensor(dep_gt['depth'].copy()).to(_DEVICE)

    for i in trange(start, N_iters):
        if not args.no_batching:
            try:
                if(args.depth_loss):
                    batch, batch_depth = get_next_batch(args, raysRGB_iter, raysDepth_iter, _DEVICE)
                else:
                    batch, _ = get_next_batch(args, raysRGB_iter, None, _DEVICE)
            except StopIteration:
                raysRGB_iter = init_dataloader(rays_rgbs, RayDataset, batch_n=args.N_rgb, device=_DEVICE)
                if(args.depth_loss):
                    raysDepth_iter = init_dataloader(rays_depths, RayDataset, batch_n=args.N_depth, device=_DEVICE)

            batch_rays, target_s = batch[:2], batch[2]
            ray_coords = batch[3,:,:2]

            if(args.depth_loss):
                batch_rays_depth = batch_depth[:2] # 2 x B x 3
                target_depth = batch_depth[2,:,0] # B
                batch_rays = torch.cat([batch_rays, batch_rays_depth], 1) # (2, 2 * N_rand, 3)
        else:
            # Sample from a single image
            img_i = np.random.choice(i_train)
            pose = pose_param_net(img_i) if args.pose_refine else poses[img_i, :3,:4]
            bbox = bboxes[img_i] if args.block_bg else []
            depth_ori_img = depth_ori[img_i] if args.smooth_loss else None
            batch_rays, target_s, target_depth, sel_coords, depth_sel_inds = sample_single_img(args, images[img_i], depth_gts[img_i], pose, intrinsics[img_i], bbox, depth_ori_img)
        
        N_batch = target_s.shape[0]
        rgb, disp, acc, depth, extras = render(H, W, None, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        if args.depth_loss:
            rgb = rgb[:N_batch, :]
            disp, dip_sup = disp[:N_batch], disp[N_batch:]
            acc = acc[:N_batch]
            depth, depth_sup = depth[:N_batch], depth[N_batch:]
            extras = {x:extras[x][:N_batch] for x in extras}
        
    #*  ---------------------=====| Loss |=====------------------------- *# 

        optimizer.zero_grad()
        if(args.pose_refine):
            optimizer_pose.zero_grad()
        if(args.depth_conf):
            optimizer_conf.zero_grad()

        img_loss = img2mse(rgb, target_s)
        loss = img_loss

        if(args.depth_loss):
            target_depth = target_depth.squeeze(-1)
            depth_loss = (depth_sup - target_depth) ** 2
            if(args.depth_conf and args.no_batching):
                base_coords, base_depends, repj_depends = select_conf_depends(args, img_i, images, depth_gts, poses, intrinsics, i_train, pose_param_net)
                conf_map = Conf(base_coords, base_depends, repj_depends, img_i)
                conf = conf_map[sel_coords[:,1], sel_coords[:,0]]
                depth_loss *= conf
                depth_loss = depth_loss[depth_loss!=0]
            
            depth_loss = torch.mean(depth_loss)
            loss += args.depth_lambda * depth_loss

        if(args.smooth_loss):
            target_dep_rgb = images[img_i][sel_coords[:, 1], sel_coords[:, 0]]
            ori_rgb = target_dep_rgb.view(args.N_patch, args.patch_sz, args.patch_sz, -1)
            ori_disp = (1/torch.clamp(depth_sup, min=1e-5))
            ori_disp = ori_disp.view(args.N_patch, args.patch_sz, args.patch_sz, -1)
            smooth_loss = edge_aware_loss_v2(ori_rgb, ori_disp)
            loss += args.smooth_lambda * smooth_loss

        psnr += mse2psnr(img_loss)

        # Course Network Loss
        if 'rgb0' in extras and not args.no_coarse:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss += img_loss0
            psnr0 = mse2psnr(img_loss0)
            
        loss.backward()

        optimizer.step()
        if(args.pose_refine):
            optimizer_pose.step()
        if(args.depth_conf):
            optimizer_conf.step()


    #*  ---------------------=====| Logs |=====------------------------- *# 

        writer.add_scalar('img_loss', img_loss, i)
        writer.add_scalar('psnr', mse2psnr(img_loss), i)
        if args.depth_loss:
            writer.add_scalar('depth_loss',depth_loss)

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if i%args.i_testset==0 and i > 0 and len(i_test) > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)

            with torch.no_grad():
                rgbs, disps = render_path(H, W, poses[i_test], intrinsics[i_test], args.chunk, render_kwargs_test)
            print('Saved test set')

            if(args.psnr_method == 'bbox'):
                psnr, ssim, lpips = calc_bbox_benchmark(images[i_test], rgbs, bboxes[i_test])
                psnr = np.array(psnr)
                ssim = np.array(ssim)
                lpips = np.array(lpips)
                # ssim(), lpips()
            elif(args.psnr_method == 'mask'):
                psnr = calc_mask_benchmark(images[i_test], rgbs, test_mask)

                rgbs[~test_mask] = 0
                rgb_p = []
                for idx, (x1, y1, x2, y2) in enumerate(bboxes[i_test].round().astype(int)):
                    rgb_crop = rgbs[idx, y1:y2, x1:x2]
                    rgb_p.append(rgb_crop)
                rgbs = np.stack(rgb_p)
            else:
                raise NotImplementedError
            
            print(f"TEST PSNR: {psnr.mean()} | SSIM: {ssim.mean()} | LPIPS: {lpips.mean()}" )
            for idx, test_im in enumerate(rgbs):
                imageio.imwrite(f'{testsavedir}/{psnr[idx]}.png', test_im)
            if(args.pose_refine):
                pose_param_net.train()

        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict() if render_kwargs_train['network_fn'] is not None else None,
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() if render_kwargs_train['network_fine'] is not None else None,
                'pose_refine_state_dict': pose_param_net.state_dict() if args.pose_refine else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()/args.i_print} lr: {new_lrate}")
            psnr = 0.

        global_step += 1

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()

