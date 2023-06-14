import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.utils.data.distributed as data_dist
import utils.render_utils as utils
from utils.sample_utils import sample_rays, sample_single_img
import collections
Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))

class PatchRayDataset(data.Dataset):
    def __init__(self, ray_data, N_patch, patch_sz):
        super(PatchRayDataset, self).__init__()

        self.N_patch = N_patch
        self.patch_sz = patch_sz
        self.length = len(ray_data)
        self.rayData = ray_data

    def sample_patches(self, rays):
        H, W, N, C = rays.shape
        if(rays.shape[0]*rays.shape[1] <= (self.patch_sz*2)**2):
            return rays.reshape(-1, N, C)
        H_crop, W_crop = H - self.patch_sz - 1, W - self.patch_sz - 1
        patch_ray_list = []
        for _ in range(self.N_patch):
            random_center = np.random.randint((self.patch_sz, self.patch_sz),(H_crop, W_crop), size=2)
            # top_left and bottom_right x: height / y: width
            (y1,x1),(y2,x2) = random_center-self.patch_sz, random_center+self.patch_sz 
            patch_rays = rays[y1:y2, x1:x2].reshape(-1, N, C)
            patch_ray_list.append(patch_rays)
        patch_ray_batch = np.stack(patch_ray_list).reshape(-1, N, C)

        return patch_ray_batch

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rays = self.sample_patches(self.rayData[index])
        return rays, index

class ImageRayDataset(data.Dataset):
    def __init__(self, ray_data, batch_n):
        super(ImageRayDataset, self).__init__()
        self.length = len(ray_data)
        self.rayData = ray_data
        self.batch_n = batch_n

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_rays = self.rayData[index]
        idx = np.linspace(0, image_rays.shape[0]-1, image_rays.shape[0]).astype(np.int)
        tgt_idx = np.random.choice(idx, self.batch_n)
        return image_rays[tgt_idx]

class RayDataset(data.Dataset):
    def __init__(self, ray_data):
        super(RayDataset, self).__init__()
        self.length = ray_data.shape[0]
        self.rayData = ray_data

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.rayData[index]
        
def init_dataloader(rays, DataSet, **kwargs):
    if(DataSet is RayDataset):
        return iter(
            DataLoader(
                RayDataset(rays),
                batch_size=kwargs['batch_n'],
                shuffle=True,
                num_workers=0,
                generator=torch.Generator(device=kwargs['device'])))
    elif(DataSet is PatchRayDataset):
        return iter(
            DataLoader(
                PatchRayDataset(rays, kwargs['N_patch'], kwargs['patch_sz']),
                batch_size=1,
                shuffle=False,
                num_workers=0,
                generator=torch.Generator(device=kwargs['device'])))
    elif(DataSet is ImageRayDataset):
        return iter(
            DataLoader(
                ImageRayDataset(rays, kwargs['batch_n']),
                batch_size=1,
                shuffle=False,
                num_workers=0,
                generator=torch.Generator(device=kwargs['device'])))
            
    else:
        print('Unkown Dataset')
        return -1


def get_next_batch(args, raysRGB_iter, raysDepth_iter, device):
    batch = next(raysRGB_iter)

    if(not args.random_sample):
        batch, img_idx = batch
    batch = batch.to(device)
    batch = batch.view(-1, 4, 3)
    batch = batch.permute(1,0,2)
    

    batch_depth = next(raysDepth_iter)
        
    batch_depth = batch_depth.to(device)
    batch_depth = batch_depth.view(-1,3,3)
    batch_depth = batch_depth.permute(1,0,2)
    return batch, batch_depth


## Implement FullDataset

class SingleImage(data.Dataset):
    def __init__(self,args, images, i_train, poses, intrinsics, depth_gts,near,far,camera_index,batch_n=1, device=None):
        super(SingleImage, self).__init__()
        self.length = len(i_train)
        self.i_train=i_train
        self.images=images
        self.depth_gts=depth_gts
        self.poses=poses
        self.intrinsics=intrinsics
        self.near=near
        self.far=far
        self.args=args
        # self.args.N_rgb=batch_n
        self.device=device

        self.batch_n = batch_n
        self.camera_index = camera_index
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_i=self.i_train[index]
        # test semantics
        # img_i = 72
        batch_rays, target_s, target_depth, sel_coords,sel_inds = sample_single_img(self.args, torch.tensor(self.images[img_i]).to(self.device),
        torch.tensor(self.depth_gts[img_i]).to(self.device),self.poses[img_i], self.intrinsics[img_i],
            self.near,self.far,near_far=self.args.near_far,batch_n=self.batch_n,app=self.camera_index[img_i])
        return  batch_rays, target_s, target_depth, sel_coords,sel_inds,img_i

def NuscenesDataLoader(args, images, i_train, poses, intrinsics, depth_gts,near,far,camera_index,batch_n=1, device=None):
   
    if not args.no_batching:
        train_dataset=FullDataset(args, images, i_train, poses, intrinsics,depth_gts,near,far)
        # sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
        if args.debug:
            return DataLoader(
                    train_dataset,
                    batch_size=batch_n,
                    shuffle=True,
                    num_workers=0,
                    generator=torch.Generator(device=device),
                    # sampler=sampler
                    )
        else:
            return DataLoader(
                        train_dataset,
                        batch_size=batch_n,
                        shuffle=True,
                        num_workers=0,
                        # sampler=sampler
                        # generator=torch.Generator(device=device)
                        )

    else:
        train_dataset=SingleImage(args, images, i_train, poses, intrinsics,depth_gts,near,far,camera_index,batch_n=batch_n)
        # sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
        if args.debug:
             return DataLoader(
                        train_dataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=0,
                        # sampler=sampler,
                        generator=torch.Generator(device=device)
                            )
        else:
            return DataLoader(
                            train_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0,
                            # sampler=sampler,
                            generator=torch.Generator(device=device)
                                )
        
                
class FullDataset(data.Dataset):
    def __init__(self, args, images, i_train, poses, intrinsics, depth_gts,near,far):
        super(FullDataset, self).__init__()
        self.images=images[i_train]
        H,W=images.shape[1:3]
        self.h=H
        self.w=W
        self.ndc=not args.no_ndc
        self.depth_loss=args.depth_loss
        self.depth=depth_gts[i_train]
        # self.mask=mask[i_train]
        self.focal = intrinsics[i_train,0,0]
        self.cx = intrinsics[i_train,0,2]
        self.cy = intrinsics[i_train,1,2]
        self.camtoworlds =poses[i_train]
        self.bd_factor=args.bds_factor
        if self.ndc:
            self.near= 0.
            self.far= 1.
        else:
            if self.bd_factor != 0.:
                self.near=near
                self.far=far
            else:
                self.near=near
                self.far=far

        self.bd_factor=args.bds_factor
       
        self._train_init()
        # needs for rays: depth,origins,directions,viewdirs,radii,near,far

    
    def _train_init(self):
        # self._load_renderings()
        self._generate_rays()
        self.images = self.images.reshape([-1, 3])
        self.depth=self.depth.reshape([-1,1])
        # self.mask=self.mask.reshape([-1,1])
        # import pdb;pdb.set_trace()
        self.rays = utils.namedtuple_map(lambda r: r.reshape([-1, r.shape[-1]]),
                                            self.rays)
    def __getitem__(self, index):
        ## for test
        # index=100;
        pixels = self.images[index]
        rays = utils.namedtuple_map(lambda r: r[index], self.rays)
        depth = self.depth[index]
        # mask = self.mask[index]
        if not self.depth_loss:
            return {"pixels": pixels, "rays": rays}
        else:
            return {"pixels": pixels, "rays": rays, 'depth':depth}
    def _generate_rays(self):
        i, j = torch.meshgrid(torch.linspace(0, self.w-1, self.w), torch.linspace(0, self.h-1, self.h))
        i=i.t().to(self.images.device);j=j.t().to(self.images.device)

        num=self.focal.shape[0]
        x=torch.broadcast_to(i,[num,self.h,self.w])
        y=torch.broadcast_to(j,[num,self.h,self.w])
        f=torch.broadcast_to(self.focal[:,None,None],[num,self.h,self.w])
        camera_dirs = torch.stack(
            [(x - self.cx[:,None,None] + 0.5) / f,
            -(y - self.cy[:,None,None] + 0.5) / f, -torch.ones_like(x)],
            axis=-1)
        ### camera_dirs.shape:[n,h,w,3]
        # directions = ((camera_dirs[None, ..., None, :] *
        #                self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
        ### directions.shape:[n,h,w,3]
        directions = ((camera_dirs[:,..., None, :] *
                    self.camtoworlds[:,None, None, :3, :3]).sum(axis=-1)).to(torch.float32)
        origins =torch.broadcast_to(self.camtoworlds[:,None, None, :3, -1],
                                directions.shape)
        viewdirs = directions / torch.linalg.norm(directions, axis=-1, keepdims=True)
        
        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = torch.sqrt(
            torch.sum((directions[:-1, :, :] - directions[1:, :, :])**2, -1))
        dx = torch.cat([dx, dx[-2:-1, :]], 0)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.

        radii = dx[..., None] * 2 / np.sqrt(12)

        ones = torch.ones_like(origins[..., :1],device=origins.device)
        
        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=ones,
            near=torch.broadcast_to(self.near[:,None,None,None],ones.shape),
            far=torch.broadcast_to(self.far[:,None,None,None],ones.shape))
        if self.ndc:
            ndc_origins, ndc_directions = convert_to_ndc(self.rays.origins,
                                                        self.rays.directions,
                                                        self.focal, self.cx, self.cy)

            mat = ndc_origins
            # Distance from each unit-norm direction vector to its x-axis neighbor.
            dx = np.sqrt(np.sum((mat[:, :-1, :, :] - mat[:, 1:, :, :]) ** 2, -1))
            dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

            dy = np.sqrt(np.sum((mat[:, :, :-1, :] - mat[:, :, 1:, :]) ** 2, -1))
            dy = np.concatenate([dy, dy[:, :, -2:-1]], 2)
            # Cut the distance in half, and then round it out so that it's
            # halfway between inscribed by / circumscribed about the pixel.
            radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)

            ones = torch.ones_like(ndc_origins[..., :1],device=ndc_origins.device)
            self.rays = Rays(
                origins=ndc_origins,
                directions=ndc_directions,
                viewdirs=self.rays.directions,
                radii=radii,
                lossmult=ones,
                near=ones * self.near,
                far=ones * self.far)
    def __len__(self):
        return self.images.shape[0]
def convert_to_ndc(origins, directions, focal, cx,cy, near=1.):
  """Convert a set of rays to NDC coordinates."""
  # Shift ray origins to near plane

  t = -(near + origins[..., 2]) / directions[..., 2]
  
  origins = origins + t[..., None] * directions

  dx, dy, dz = tuple(torch.moveaxis(directions, -1, 0))
  ox, oy, oz = tuple(torch.moveaxis(origins, -1, 0))

  # Projection
#   o0 = -((2 * focal) / w) * (ox / oz)
#   o1 = -((2 * focal) / h) * (oy / oz)
#   o2 = 1 + 2 * near / oz
  o0 = -(focal/cx)[:,None,None] * (ox / oz)
  o1 = -(focal/cy)[:,None,None] * (oy / oz)
  o2 = 1 + 2 * near / oz
#   o0 = -(2* focal/w)[:,None,None] * (ox / oz)
#   o1 = -(2 * focal/h)[:,None,None] * (oy / oz)
#   o2 = 1 + 2 * near / oz

#   d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
#   d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
#   d2 = -2 * near / oz
  d0 = -(focal/cx)[:,None,None] * (dx / dz - ox / oz)
  d1 = -(focal/cy)[:,None,None] * (dy / dz - oy / oz)
  d2 = -2 * near / oz
#   d0 = -((2 * focal) / w)[:,None,None] * (dx / dz - ox / oz)
#   d1 = -((2 * focal) / h)[:,None,None] * (dy / dz - oy / oz)
#   d2 = -2 * near / oz
  origins = torch.stack([o0, o1, o2], -1)
  directions = torch.stack([d0, d1, d2], -1)
  return origins, directions