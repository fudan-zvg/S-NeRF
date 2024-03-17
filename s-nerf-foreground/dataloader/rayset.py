import torch.utils.data as data
from torch.utils.data import DataLoader
from utils.sample_utils import sample_patches
import numpy as np
import torch

class PatchRayDataset(data.Dataset):
    def __init__(self, ray_data, N_patch, patch_sz):
        super(PatchRayDataset, self).__init__()

        self.N_patch = N_patch
        self.patch_sz = patch_sz
        self.length = len(ray_data)
        self.rayData = ray_data


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rays = sample_patches(self.rayData[index], self.patch_sz, self.N_patch)
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
    batch_depth = None
    if(not args.random_sample):
        batch, img_idx = batch
    batch = batch.to(device)
    batch = batch.view(-1, 4, 3)
    batch = batch.permute(1,0,2)

    if(args.depth_loss):
        batch_depth = next(raysDepth_iter)
        
        batch_depth = batch_depth.to(device)
        batch_depth = batch_depth.view(-1,3,3)
        batch_depth = batch_depth.permute(1,0,2)
    return batch, batch_depth