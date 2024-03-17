import torch
import numpy as np
from model.run_nerf_helpers import img2mse, mse2psnr
import lpips
import imageio
from utils.pytorch_msssim import ssim

'''
calculate the psnr
Params:
    images: target gt img
    rgbs: render rgb
Returns:
    test_psnr: torch.tensor [float]
'''

def calc_bbox_benchmark(images, rgbs, bbox_test, save=False):
    bbox_test = bbox_test.round().astype(np.int)
    rgb_list = []
    img_list = []
    calc_lpips = lpips.LPIPS(net='alex')
    for idx, (x1, y1, x2, y2) in enumerate(bbox_test):
        rgb_crop = torch.Tensor(rgbs[idx, y1:y2, x1:x2]).clone()
        img_crop = images[idx, y1:y2, x1:x2].clone()
        rgb_list.append(rgb_crop)
        img_list.append(img_crop)
    if(save):
        imageio.imwrite('fake_img.png', rgb_list[0].cpu())
        imageio.imwrite('real_img.png', img_list[0].cpu())
    psnrs = []
    ssims = []
    lpips_list = []
    for rgb, img in zip(rgb_list, img_list):
        test_loss = img2mse(rgb, img)
        test_psnr = mse2psnr(test_loss).item()
        test_ssim = ssim(rgb[None,...].float().permute(0,3,1,2), img[None,...].float().permute(0,3,1,2), full=False).item()
        test_lpips = calc_lpips.forward(rgb[None,...].float().permute(0,3,1,2), img[None,...].float().permute(0,3,1,2)).item()
        psnrs.append(test_psnr)
        ssims.append(test_ssim)
        lpips_list.append(test_lpips)
    return psnrs, ssims, lpips_list


def calc_mask_psnr(images, rgbs, masks):
    rgbs = torch.tensor(rgbs)
    masks = torch.tensor(masks)

    test_loss = img2mse(images[masks], rgbs[masks])
    test_psnr = mse2psnr(test_loss)
    return test_psnr


