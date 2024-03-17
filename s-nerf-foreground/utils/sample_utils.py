import numpy as np
import torch
from model.run_nerf_helpers import get_rays_np, get_rays_np_bbox, get_rays_by_coord_np, get_rays_by_coord

def sample_full_img(args, images, i_train, poses, intrinsics):
    H, W = images[0].shape[:2]
    rays = np.stack([get_rays_np(H, W, poses[:,:3,:4][i], intrinsics[i]) for i in i_train], 0) # [N, ro+rd, H, W, 3]
    if args.debug:
        print('rays.shape:', rays.shape)
    print('done, concats')
    rays_rgb = np.concatenate([rays, images[i_train, None]], 1) # [N, ro+rd+rgb, H, W, 3]
    if args.debug:
        print('rays_rgb.shape:', rays_rgb.shape)
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    return rays_rgb


def sample_crop_img(args, images, i_train, bboxes, poses, intrinsics):
    bboxes_int = bboxes.round().astype(np.int)
    rays_list = []
    coords_list = []
    rays_rgb = None
    for i in i_train:
        rays_o, rays_d, coords = get_rays_np_bbox(poses[:,:3,:4][i], intrinsics[i], bboxes[i])
        rays = np.stack([rays_o, rays_d])
        rays_list.append(rays)
        coords_list.append(coords)
        
    img_list = [img[bbox[1]:bbox[3], bbox[0]:bbox[2]] for img, bbox in zip(images[i_train], bboxes_int[i_train])]
    rays_rgb_list = [np.concatenate([rays, imgs[None, :], coords[None, :]], 0) for rays, imgs, coords in zip(rays_list, img_list, coords_list)] # N x [ro+rd+rgb, H_idx, W_idx, 3]
    rays_rgb_list = [np.transpose(ray, [1, 2, 0, 3]) for ray in rays_rgb_list]
    if(args.random_sample):
        rays_rgb_list = [np.reshape(ray, [-1,4,3]).astype(np.float32) for ray in rays_rgb_list] # N x [H*W, ro+rd+rgb, 3]
        rays_rgb = np.concatenate(rays_rgb_list, axis=0)
    else:
        rays_rgb = [rays_rgb for rays_rgb in rays_rgb_list]
    return rays_rgb

def sample_depth(args, depth_gts, i_train, poses, intrinsics):
    rays_depth_list = []
    rays_depth = None
    for i in i_train:
        rays_depth = np.stack(
                get_rays_by_coord_np(poses[i,:3,:4], depth_gts[i]['coord'], intrinsics[i]), 
                axis=0) # 2 x N x 3

        rays_depth = np.transpose(rays_depth, [1,0,2])
        depth_value = np.repeat(depth_gts[i]['depth'][:,None,None], 3, axis=2) # N x 1 x 3
        rays_depth = np.concatenate([rays_depth, depth_value], axis=1) # N x 4 x 3
        rays_depth_list.append(rays_depth.astype(np.float32))
        
        if(args.random_sample):
            rays_depth = np.concatenate(rays_depth_list, axis=0)
        else:
            rays_depth = [rays_depth for rays_depth in rays_depth_list]
    return rays_depth

def sample_patches(rays, patch_sz, N_patch):
    H, W, N, C = rays.shape

    H_crop, W_crop = H - patch_sz//2 - 1, W - patch_sz//2 - 1
    patch_ray_list = []
    for _ in range(N_patch):
        random_center = np.random.randint((patch_sz//2, patch_sz//2),(H_crop, W_crop), size=2)
        # top_left and bottom_right x: height / y: width
        (y1,x1),(y2,x2) = random_center-patch_sz//2, random_center+patch_sz//2
        patch_rays = rays[y1:y2, x1:x2].reshape(-1, N, C)
        patch_ray_list.append(patch_rays)
    patch_ray_batch = np.stack(patch_ray_list).reshape(-1, N, C)

    return patch_ray_batch

def sample_patches_pt(rays, patch_sz, N_patch, coords):
    H, W, C = rays.shape

    H_crop, W_crop = H - patch_sz//2 - 1, W - patch_sz//2 - 1
    patch_ray_list = []
    random_idx = np.random.randint(coords.shape[0], size=N_patch)
    random_centers = coords[random_idx]
    # top_left and bottom_right x: height / y: width
    coord_lt = random_centers-patch_sz//2
    coord_rb = random_centers+patch_sz//2
    sel_pat_coord = torch.cat([coord_lt, coord_rb], 1)
    for y1,x1,y2,x2 in sel_pat_coord:
        patch_rays = rays[y1:y2, x1:x2].reshape(-1, C)
        patch_ray_list.append(patch_rays)
    patch_ray_batch = torch.stack(patch_ray_list).reshape(-1, C)
    
    return patch_ray_batch


def sample_single_img(args, image, depth_gt, pose, intrinsic, bbox, depth_ori):
    H, W = image.shape[:2]
    x1, y1, x2, y2 = bbox.round().astype(np.int).tolist()
    if(len(bbox)):
        i, j = torch.meshgrid(torch.linspace(x1, x2, x2-x1+1), torch.linspace(y1, y2, y2-y1+1))
    else:
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))

    rgb_coords = torch.stack([i, j], -1).reshape(-1, 2) # H*W, N
    rgb_sel_inds = np.random.choice(rgb_coords.shape[0], size=[args.N_rgb], replace=False)
    sel_rgb_coords = rgb_coords[rgb_sel_inds].long()

    if(args.smooth_loss):
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
        depth_coords = torch.stack([i, j], -1) 
        sel_dep_coords = sample_patches_pt(depth_coords, args.patch_sz, args.N_patch, depth_gt['coord']).long()
        depth_sel_inds = sel_dep_coords
    else:
        depth_coords = depth_gt['coord']
        depth_sel_inds = np.random.choice(depth_coords.shape[0], size=[args.N_depth], replace=False)
        sel_dep_coords = depth_coords[depth_sel_inds].long()

    rgb_rays = torch.stack(
            get_rays_by_coord(pose, sel_rgb_coords, intrinsic), 
            axis=0) # 2 x N x 3
    dep_rays = torch.stack(
            get_rays_by_coord(pose, sel_dep_coords, intrinsic), 
            axis=0) # 2 x N x 3

    batch_rays = torch.cat([rgb_rays, dep_rays], 1)
    
    target_rgb = image[sel_rgb_coords[:, 1], sel_rgb_coords[:, 0]].detach()
    target_dep = depth_gt['depth'][depth_sel_inds] if not args.smooth_loss else depth_ori[sel_dep_coords[:,1], sel_dep_coords[:,0]].detach()
    
    return batch_rays, target_rgb, target_dep, sel_dep_coords, depth_sel_inds

def sample_test_img(N_rays, image, pose, intrinsic, bbox):
    H, W = image.shape[:2]
    x1, y1, x2, y2 = bbox.round().astype(np.int).tolist()
    if(len(bbox)):
        i, j = torch.meshgrid(torch.linspace(x1, x2, x2-x1+1), torch.linspace(y1, y2, y2-y1+1))
    else:
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))

    rgb_coords = torch.stack([i, j], -1).reshape(-1, 2) # H*W, N
    rgb_sel_inds = np.random.choice(rgb_coords.shape[0], size=[N_rays], replace=False)
    sel_rgb_coords = rgb_coords[rgb_sel_inds].long()
        

    rgb_rays = torch.stack(
            get_rays_by_coord(pose, sel_rgb_coords, intrinsic), 
            axis=0) # 2 x N x 3

    batch_rays = torch.cat([rgb_rays], 1)
    target_rgb = image[sel_rgb_coords[:, 1], sel_rgb_coords[:, 0]].detach()
    return batch_rays, target_rgb



def sample_rays(args, images, i_train, poses, intrinsics, depth_gts=None, bboxes=None):
    rays_depths = None
    if(not args.block_bg):
        rays_rgbs = sample_full_img(args, images, i_train, poses, intrinsics)
    else:
        rays_rgbs = sample_crop_img(args, images, i_train, bboxes, poses, intrinsics)
        
    if(args.depth_loss):
        rays_depths = sample_depth(args, depth_gts, i_train, poses, intrinsics)
    
    return rays_rgbs, rays_depths


