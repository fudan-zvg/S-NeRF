from tqdm import tqdm
import os
import numpy as np

from torchvision import transforms as T

import torch
import yaml

img_size = (640, 960)
if_resize = 0
lama_or_AOT = 1 # 1 for AOT, 0 for lama
root = "../waymo_scene4/images/"
output_root = "../mvobj_waymo_scene4_ft"

kenelsize = 50
num_imgs = 0

pre_train = os.path.join(os.path.dirname(__file__), "AOT-GAN-for-Inpainting/checkpoint/G0000000.pt")

transform_mask = T.Compose([
    T.ToTensor(),
])

transform_in = T.Compose([T.Resize(img_size),T.ToTensor(), T.ColorJitter(0.05, 0.05, 0.05, 0.05)]) if if_resize \
    else T.Compose([T.ToTensor(), T.ColorJitter(0.05, 0.05, 0.05, 0.05),
])
transform_in_nojitter = T.Compose([T.ToTensor()])

def init_inpainting_AOT(device):
    import importlib
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "AOT-GAN-for-Inpainting/src"))
    sys.path.append(os.path.dirname(__file__))
    net = importlib.import_module('model.'+'aotgan')

    from AOT_option import args
    model = net.InpaintGenerator(args).cuda()
    # import pdb; pdb.set_trace()

    model.load_state_dict(torch.load(pre_train, map_location='cuda'))
    model.eval()
    return model

def create_mask(label_list, pred, kernelsize=10):
    mask_all = 1-np.where(pred==label_list[0], 255, 0).astype('uint8')
    for label in label_list:
        mask_all = mask_all*(1-np.where(pred==label, 255, 0).astype('uint8'))

    mask = 1-mask_all

    import cv2 as cv
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernelsize, kernelsize))

    mask = cv.dilate(mask, kernel)

    return mask


def inpainting_AOT(img, mask, model, device, light=False):
    # img = Image.open(img_path).convert('RGB')
    img = transform_in_nojitter(img).unsqueeze(0)  # To tensor of NCHW
    img = img.to(device)

    img = img*2-1.0

    mask = transform_mask(mask).unsqueeze(0)
    mask = mask.to(device)

    image_masked = img * (1 - mask.float()) + mask
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        pred_img = model(image_masked, mask)

    def postprocess(image):
        image = torch.clamp(image, -1., 1.)
        image = (image + 1) / 2.0
        return image

    comp_imgs = (1 - mask) * img + mask * pred_img

    img_inpaint = postprocess(comp_imgs[0])
    img = postprocess(img)
    image_masked = postprocess(image_masked)
    return img_inpaint, mask, image_masked



def init_inpaiting(device, light=False):
    from omegaconf import OmegaConf
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'lama'))
    sys.path.append(os.path.dirname(__file__))
    from lama.saicinpainting.training.trainers import DefaultInpaintingTrainingModule



    train_config_path = os.path.join(os.path.dirname(__file__), 'lama/lama-fourier/config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    train_config.training_model.predict_only = True
    kwargs = dict(train_config.training_model)
    kwargs.pop('kind')
    kwargs['use_ddp'] = True
    # kwargs['use_ddp'] = False
    inpaint_model = DefaultInpaintingTrainingModule(train_config, **kwargs)
    if not light:
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'lama/lama-fourier/models/best.ckpt')
    else:
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'lama/lama-fourier/models/light.ckpt')
    state = torch.load(checkpoint_path, map_location='cpu')
    inpaint_model.load_state_dict(state['state_dict'], strict=False)
    inpaint_model.to(device)
    inpaint_model.eval()
    return inpaint_model


def inpainting(img, mask, model, device, light=False):

    # img = Image.open(img_path).convert('RGB')
    img = transform_in_nojitter(img).unsqueeze(0)  # To tensor of NCHW
    img = img.to(device)
    # img = img/255.0  ###!!!


    mask = transform_mask(mask).unsqueeze(0)
    mask = mask.to(device)
    # mask = mask/255

    if not light:
        img = img*(1-mask)

    batch = dict(image=img, mask=mask)
    batch = model(batch)
    img_inpaint = batch['inpainted']
    return img_inpaint, mask, img


def choose_model_inpaint(imgs, masks, lama_or_AOT=1,
                         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                         light=False):
    '''
    imgs: list of img, img: numpy, HWC
    masks: list of mask, mask: 0-255 HW
    lama_or_AOT: choose model, 1 for AOT, 0 for lama
    '''
    inpainting_model_list = [init_inpaiting, init_inpainting_AOT]
    inpainting_list = [inpainting, inpainting_AOT]

    inpainting_model = inpainting_model_list[lama_or_AOT](device, light)

    result = []
    with torch.no_grad():
        for img, mask in tqdm(zip(imgs, masks)):
            img_inpaint, mask, img_ori = inpainting_list[lama_or_AOT](img, mask, inpainting_model, device, light)
            result.append(img_inpaint)

    return result