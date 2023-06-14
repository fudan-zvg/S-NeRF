from importlib.resources import path
from matplotlib.pyplot import axis
import numpy as np
# from sympy import rad
import torch
import utils.render_utils as utils
from model.run_nerf_helpers import get_rays_np, get_rays_np_bbox, get_rays_by_coord_np, get_rays_by_coord, ndc_rays
import collections


Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far','app'))
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

# def sample_depth(args, depth_gts, i_train, poses, intrinsics):
#     rays_depth_list = []
#     rays_depth = None
#     for i in i_train:
#         rays_depth = np.stack(
#                 get_rays_by_coord_np(poses[i,:3,:4], depth_gts[i]['coord'], intrinsics[i]), 
#                 axis=0) # 2 x N x 3

#         rays_depth = np.transpose(rays_depth, [1,0,2])
#         depth_value = np.repeat(depth_gts[i]['depth'][:,None,None], 3, axis=2) # N x 1 x 3
#         rays_depth = np.concatenate([rays_depth, depth_value], axis=1) # N x 4 x 3
#         rays_depth_list.append(rays_depth.astype(np.float32))
        
#         if(args.random_sample):
#             rays_depth = np.concatenate(rays_depth_list, axis=0)
#         else:
#             rays_depth = [rays_depth for rays_depth in rays_depth_list]
#     return rays_depth
def sample_patches_pt(H,W,rays, patch_sz, N_patch):
    # ray is 2D coordinate
    # H, W = rays.shape
    
    patch_ray_list = []
    x_mask= (rays[:,0]>patch_sz)&(rays[:,0]<H-patch_sz)
    y_mask= (rays[:,1]>patch_sz)&(rays[:,1]<W-patch_sz)
    mask_rays=rays[x_mask & y_mask]
    random_idx = np.random.randint(mask_rays.shape[0], size=N_patch)
    random_centers = mask_rays[random_idx]
    # top_left and bottom_right x: height / y: width
    coord_lt = random_centers-patch_sz//2
    coord_rb = random_centers+patch_sz//2
    sel_pat_coord = torch.cat([coord_lt, coord_rb], 1)
    for y1,x1,y2,x2 in sel_pat_coord:
       
        patch_rays = rays.reshape(H,W,-1)[y1:y2, x1:x2]
        patch_ray_list.append(patch_rays)
    patch_ray_batch = torch.stack(patch_ray_list)
    patch_ray_batch=patch_ray_batch.reshape(-1,2)
   
    return patch_ray_batch


def sample_single_img(args, image, depth_gt, pose, intrinsic,near=0.,far=1.,near_far=False,batch_n=None,app=0.):
    H, W = image.shape[:2]
    # if bbox:
    #     x1, y1, x2, y2 = bbox.round().astype(np.int).tolist()
    
    #     i, j = torch.meshgrid(torch.linspace(x1, x2, x2-x1+1), torch.linspace(y1, y2, y2-y1+1))
    # else:
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i=i.t().to(image.device);j=j.t().to(image.device)
    rgb_coords = torch.stack([j,i], -1).long().reshape(-1, 2) # H*W, N
    if args.smooth_loss:
        smooth_rays=sample_patches_pt(H,W,rgb_coords, args.patch_sz, args.N_patch)
    cx=intrinsic[0,2];cy=intrinsic[1,2];f=(intrinsic[0,0]+intrinsic[1,1])/2
    camera_dirs = torch.stack(
        [(i - cx + 0.5) / f,
        -(j-cy+ 0.5) / f, -torch.ones_like(i)],
        axis=-1)
    ### camera_dirs.shape:[n,h,w,3]
    # directions = ((camera_dirs[None, ..., None, :] *
    #                self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
    ### directions.shape:[n,h,w,3]
    directions = ((camera_dirs[..., None, :] *
                pose[None, None, :3, :3]).sum(axis=-1))
    
    # viewdirs = directions / torch.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = torch.sqrt(
        torch.sum((directions[:-1, :, :] - directions[1:, :, :])**2, -1))
    dx = torch.cat([dx, dx[-2:-1, :]], 0)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.

    radii = dx[..., None] * 2 / np.sqrt(12)

    # depth_coords = depth_gt['coord']
    if batch_n is not None:
        rgb_sel_inds = np.random.choice(rgb_coords.shape[0], size=[batch_n], replace=False)
    else:
        rgb_sel_inds = np.random.choice(rgb_coords.shape[0], size=[args.N_rgb], replace=False)
    # depth_sel_inds = np.random.choice(depth_coords.shape[0], size=[args.N_depth], replace=False)
    
    sel_rgb_coords = rgb_coords[rgb_sel_inds].long()
    if args.smooth_loss:
        # at this time, the final coords will be the smooth rays
        sel_rgb_coords =torch.cat([sel_rgb_coords,smooth_rays])
    # sel_dep_coords = depth_coords[depth_sel_inds].long()
    
    rgb_rays = torch.stack(
            get_rays_by_coord(pose, sel_rgb_coords, intrinsic), 
            axis=0) # 2 x N x 3
    rays_o,rays_d=rgb_rays.float()
   
    # dep_rays = torch.stack(
    #         get_rays_by_coord(pose, sel_dep_coords, intrinsic), 
    #         axis=0) # 2 x N x 3

    # batch_rays = torch.cat([rgb_rays, dep_rays], 1)
    
    
    
    if not args.no_ndc:
        near=0.;far=1.
        origins =torch.broadcast_to(pose[None, None, :3, -1],
                            directions.shape)
        origins, directions=ndc_rays(H, W, f, 1., origins, directions)
        mat = directions
        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = torch.sqrt(torch.sum((mat[:-1, :, :] - mat[1:, :, :]) ** 2, -1))
        dx = torch.cat([dx, dx[-2:-1, :]], 0)

        dy = torch.sqrt(torch.sum((mat[:, :-1, :] - mat[:, 1:, :]) ** 2, -1))
        dy = torch.cat([dy, dy[:, -2:-1]], 1)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)
        rays_o=origins.reshape([-1,3])[rgb_sel_inds].reshape([-1,3])
        rays_d=directions.reshape([-1,3])[rgb_sel_inds].reshape([-1,3])
    else:
        # set a fixed near and bound for all rays.
        if not near_far:
            near = near*0.9
            far = far*1.1
        else:
            near = depth_gt[depth_gt!=0].min()*0.9
            far = depth_gt[depth_gt!=0].max()*1.1
        # import pdb;pdb.set_trace()
        # print('near:',near)
        # print('far',far)
    radii=radii[sel_rgb_coords[:,0],sel_rgb_coords[:,1]].reshape([-1,1])
    # radii= radii.reshape([-1,1])[rgb_sel_inds].reshape([-1,1])
    # rays_o=origins.reshape([-1,3])[rgb_sel_inds].reshape([-1,3])
    # rays_d=directions.reshape([-1,3])[rgb_sel_inds].reshape([-1,3])
    # viewdirs=viewdirs.reshape([-1,3])[rgb_sel_inds].reshape([-1,3])
    viewdirs=rays_d/ torch.linalg.norm(rays_d, axis=-1, keepdims=True)
    ones = torch.ones_like(rays_o[..., :1],device=rays_o.device)
    # if not args.encode_appearance:
    #     batch_rays = Rays(
    #         origins=rays_o,
    #         directions=rays_d,
    #         viewdirs=viewdirs,
    #         radii=radii,
    #         lossmult=ones,
    #         near=ones * near,
    #         far=ones * far)
    # else:
    batch_rays = Rays(
        origins=rays_o,
        directions=rays_d,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=ones,
        near=ones * near,
        far=ones * far,
        app=ones*torch.tensor(app))
    
    target_rgb = image[sel_rgb_coords[:, 0], sel_rgb_coords[:, 1]]
    target_dep = depth_gt[sel_rgb_coords[:, 0], sel_rgb_coords[:, 1]]
   
    return batch_rays, target_rgb, target_dep, sel_rgb_coords,rgb_sel_inds


def sample_rays(args, images, i_train, poses, intrinsics, depth_gts=None, bboxes=None):
    rays_depths = None
    if(not args.block_bg):
        rays_rgbs = sample_full_img(args, images, i_train, poses, intrinsics)
        import pdb; pdb.set_trace()
    else:
        rays_rgbs = sample_crop_img(args, images, i_train, bboxes, poses, intrinsics)
        
    rays_depths = sample_depth(args, depth_gts, i_train, poses, intrinsics)
        
    return rays_rgbs, rays_depths




def get_rays_panorama(args,H,W,pose,intrinsic,near,far,factor=4,origin=None):
    
    H, W = H//factor, W//factor
    temp_intrinsic = intrinsic/factor
    cy=0.5*H;f=(temp_intrinsic[0,0]+temp_intrinsic[1,1])/2
    fov_x=2*torch.arctan(0.5*W/f)/np.pi*180.
    # import pdb; pdb.set_trace()
    W= int(150/fov_x*W)
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i=i.t();j=j.t()
    rgb_coords = torch.stack([j,i], -1).reshape(-1, 2) # H*W, N
    #TODO: change the intrinsic for the panorama image
    
    # cx = [0.5*W+W*t for t in range(6)]
    camera_dirs =torch.stack(
        [(i - i) / f,
        -(j -cy+ 0.5) / f, -torch.ones_like(i)],
        axis=-1) 
    from scipy.spatial.transform import Rotation as R
    rotation_degree=np.linspace(75,-75,int(W))
    rotation = R.from_euler('y', rotation_degree, degrees='True').as_matrix().astype(np.float32)
    rotation=torch.tensor(rotation)
    directions = ((camera_dirs[..., None, :] *
                rotation[None, :, :3, :3]).sum(axis=-1))
    if origin is not None:
        origins=torch.broadcast_to(rotation[None, :, :3, -1]+origin,
                                directions.shape)
    else:
        origins =torch.broadcast_to(rotation[None, :, :3, -1],
                                directions.shape)
    viewdirs = directions / torch.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = torch.sqrt(
        torch.sum((directions[:-1, :, :] - directions[1:, :, :])**2, -1))
    dx = torch.cat([dx, dx[-2:-1, :]], 0)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.

    radii = dx[..., None] * 2 / np.sqrt(12)
    

    near =near*0.9
    far = far*1.3
    ones = torch.ones_like(origins[..., :1],device=origins.device)
    batch_rays = Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=ones,
        near=ones * near,
        far=ones * far,
        app=ones * 0.)

    return batch_rays

def get_rays_single_img(args, image, depth_gt, pose, intrinsic,near=0.,far=1., factor=4):
    H, W = image.shape[:2]

    H, W = H//factor, W//factor
    temp_intrinsic = intrinsic/factor

    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i=i.t().to(image.device);j=j.t().to(image.device)
    rgb_coords = torch.stack([j,i], -1).reshape(-1, 2) # H*W, N
    cx=temp_intrinsic[0,2];cy=temp_intrinsic[1,2];f=(temp_intrinsic[0,0]+temp_intrinsic[1,1])/2
    camera_dirs = torch.stack(
        [(i - cx + 0.5) / f,
        -(j -cy+ 0.5) / f, -torch.ones_like(i)],
        axis=-1)

    directions = ((camera_dirs[..., None, :] *
                pose[None, None, :3, :3]).sum(axis=-1))
    origins =torch.broadcast_to(pose[None, None, :3, -1],
                            directions.shape)
    viewdirs = directions / torch.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = torch.sqrt(
        torch.sum((directions[:-1, :, :] - directions[1:, :, :])**2, -1))
    dx = torch.cat([dx, dx[-2:-1, :]], 0)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.

    radii = dx[..., None] * 2 / np.sqrt(12)
    
    
    if not args.no_ndc:
        near=0.;far=1.
        origins, directions=ndc_rays(H, W, f, 1., origins, directions)
        mat = directions
        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = torch.sqrt(torch.sum((mat[:-1, :, :] - mat[1:, :, :]) ** 2, -1))
        dx = torch.cat([dx, dx[-2:-1, :]], 0)

        dy = torch.sqrt(torch.sum((mat[:, :-1, :] - mat[:, 1:, :]) ** 2, -1))
        dy = torch.cat([dy, dy[:, -2:-1]], 1)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)
    else:
        near =near*0.9
        far = far*1.1

    ones = torch.ones_like(origins[..., :1],device=origins.device)
    batch_rays = Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=ones,
        near=ones * near,
        far=ones * far,
        app=ones * 0.)

    return batch_rays
    
def get_rays_single_img_(args, image, pose,hwf,bds, near=0.,far=1.):
    H, W = image.shape[:2]
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i=i.t().to(image.device);j=j.t().to(image.device)
    # rgb_coords = torch.stack([j,i], -1).reshape(-1, 2) # H*W, N
    cx=0.5*W;cy=0.5*H;f=hwf[-1]
    camera_dirs = torch.stack(
        [(i - cx + 0.5) / f,
        -(j -cy+ 0.5) / f, -torch.ones_like(i)],
        axis=-1)
    ### camera_dirs.shape:[n,h,w,3]
    # directions = ((camera_dirs[None, ..., None, :] *
    #                self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
    ### directions.shape:[n,h,w,3]
    directions = ((camera_dirs[..., None, :] *
                pose[None, None, :3, :3]).sum(axis=-1))
    origins =torch.broadcast_to(pose[None, None, :3, -1],
                            directions.shape)
    viewdirs = directions / torch.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = torch.sqrt(
        torch.sum((directions[:-1, :, :] - directions[1:, :, :])**2, -1))
    dx = torch.cat([dx, dx[-2:-1, :]], 0)
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.

    radii = dx[..., None] * 2 / np.sqrt(12)
    
    
    if not args.no_ndc:
        origins, directions=ndc_rays(H, W, f, 1., origins, directions)
        mat = directions
        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = torch.sqrt(torch.sum((mat[:-1, :, :] - mat[1:, :, :]) ** 2, -1))
        dx = torch.cat([dx, dx[-2:-1, :]], 0)

        dy = torch.sqrt(torch.sum((mat[:, :-1, :] - mat[:, 1:, :]) ** 2, -1))
        dy = torch.cat([dy, dy[:, -2:-1]], 1)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)
    else:
        near = bds.min()
        far = bds.max()
        # print('near:',near)
        # print('far',far)
    # radii= radii.reshape([-1,1])[rgb_sel_inds].reshape([-1,1])
    # rays_o=origins.reshape([-1,3])[rgb_sel_inds].reshape([-1,3])
    # rays_d=directions.reshape([-1,3])[rgb_sel_inds].reshape([-1,3])
    # viewdirs=viewdirs.reshape([-1,3])[rgb_sel_inds].reshape([-1,3])
    ones = torch.ones_like(origins[..., :1],device=origins.device)
    batch_rays = Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=ones,
        near=ones * near,
        far=ones * far)

    return batch_rays

def sample_rays(args, rays_iter, rays_loader, pose_param_net):
    try:
        batch_rays, target_s, target_depth,sel_coords,sel_inds,img_i = next(rays_iter)
    except StopIteration:
        rays_iter = iter(rays_loader)
        batch_rays, target_s, target_depth, sel_coords,sel_inds,img_i = next(rays_iter)
    # change the dimension
    target_s = target_s.squeeze(0)
    target_depth=target_depth[0]
    sel_coords=sel_coords[0]
    img_i=img_i.item()
    pose = torch.eye(4) if not args.pose_refine else pose_param_net(img_i,transform_only=True)
    batch_rays = utils.namedtuple_map(lambda r: r.squeeze(0),batch_rays)
    directions = (batch_rays.directions[:,None,:]*pose[:3,:3]).sum(axis=-1)
    viewdirs = (batch_rays.viewdirs[:,None,:]*pose[:3,:3]).sum(axis=-1)
    origins = batch_rays.origins+pose[:3,3]
    batch_rays = Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=batch_rays.radii,
        lossmult=batch_rays.lossmult,
        near=batch_rays.near,
        far=batch_rays.far,
        app=batch_rays.app)
    return batch_rays, sel_coords, img_i, target_s, target_depth