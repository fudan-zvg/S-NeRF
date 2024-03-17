import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.loss import get_reproj_conf
from torchvision import models

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().to(device)
        self.device = device
        self.criterion = nn.L1Loss(reduction='none')
        self.weights = [1.0/16, 1.0/8, 1.0/4, 1.0]#[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]     

    def forward(self, x, y):            
        x = x.permute(2,0,1).float()
        y = y.permute(2,0,1).float()
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = torch.zeros((900,1600)).to(self.device)
        for i in range(len(x_vgg)-1):
            feat_x, feat_y = x_vgg[i], y_vgg[i].detach()
            if(i > 0):
                feat_x = F.upsample(x_vgg[i].unsqueeze(0), mode='bilinear', size=(900, 1600), align_corners=True)
                feat_y = F.upsample(y_vgg[i].unsqueeze(0), mode='bilinear', size=(900, 1600), align_corners=True)
                feat_x = feat_x.squeeze(0)
                feat_y = feat_y.squeeze(0)

            f_loss = self.weights[i] * self.criterion(feat_x, feat_y)
            loss += f_loss.mean(0)
        return loss
        
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(21, 30):
        #     self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        #h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4] #, h_relu5]
        return out

class Confidence(nn.Module):
    def __init__(self, modes, device, images_num, vgg_loss=False):
        super(Confidence, self).__init__()        
        self.modes = modes
        # if vgg_loss:
        self.lambdas = nn.Parameter(torch.zeros((len(modes),images_num)),requires_grad=True)
        # else:
        #     self.lambdas = nn.Parameter(torch.zeros((len(modes)-1)),requires_grad=True)               
        self.vgg_loss = None
        if(vgg_loss):
            self.vgg_loss = VGGLoss(device)
    
    '''
    Save the confidence to the given path
    '''
    def precompute_conf_map(self, args,cam_index,images, depth_gts,poses,intrinsics,i_train):
        conf_maps=[]
        for i in range(len(i_train)):
            img_i=i_train[i]
            base_depends, repj_depends=select_conf_depends(args, img_i, cam_index, images, depth_gts, poses, intrinsics, i_train, pose_param_net=None, flow=None)
            conf_map = self.forward(base_depends, repj_depends,img_i,ret_dict=True)
            conf_maps.append(conf_map)
        return conf_maps
        
    '''
    Project base idx images to others to calculate confidence
    '''
    def forward(self, proj_coord, base_depends, repj_depends, img_i,ret_dict=False):
        base_img = base_depends[0]
        conf_maps = get_reproj_conf(self.modes,
                        proj_coord, 
                        base_depends, 
                        repj_depends,
                        self.vgg_loss)
        
        H, W = base_img.shape[:2]
        final_conf_map = torch.zeros((H,W))
        weights=F.sigmoid(self.lambdas[:,img_i])
        for idx, mode in enumerate(self.modes):
            final_conf_map[proj_coord[:,1], proj_coord[:,0]] += weights[idx] * conf_maps[mode]
        final_conf_map/=weights.sum()
        
        # vis_conf_map(H, W, final_conf_map.reshape(-1,1)[:,0], proj_coord, 'final')
        if not ret_dict:
            return final_conf_map
        else:
            return conf_maps

        
def select_conf_depends(args, img_i, images, depth_gts, poses, intrinsics, i_train, pose_param_net=None):
    repj_imgs = []
    repj_poses = []
    repj_intrs = []
    repj_depths = []
    base_img = images[img_i]
    base_pose = pose_param_net(img_i) if args.pose_refine else poses[img_i]
    base_intr = intrinsics[img_i]
    
    cam_idx = np.where(i_train == img_i)[0].item()
    if(cam_idx > 0):
        sel_id = i_train[cam_idx-1]
        repj_imgs.append(images[sel_id])
        repj_poses.append(pose_param_net(sel_id))
        depth_map = torch.zeros((base_img.shape[0], base_img.shape[1]))
        coords = depth_gts[sel_id]['coord']
        depth_map[coords[:,1], coords[:,0]] = depth_gts[sel_id]['depth'][:,0]
        repj_depths.append(depth_map)
        repj_intrs.append(intrinsics[sel_id])
    if(cam_idx < len(i_train)-1):
        sel_id = i_train[cam_idx+1]
        repj_imgs.append(images[sel_id])
        repj_poses.append(pose_param_net(sel_id))
        depth_map = torch.zeros((base_img.shape[0], base_img.shape[1]))
        coords = depth_gts[sel_id]['coord']
        depth_map[coords[:,1], coords[:,0]] = depth_gts[sel_id]['depth'][:,0]
        repj_depths.append(depth_map)
        repj_intrs.append(intrinsics[sel_id])

    base_coords = depth_gts[img_i]['coord']
    base_depth = torch.zeros((base_img.shape[0], base_img.shape[1]))
    base_depth[base_coords[:,1], base_coords[:,0]] = depth_gts[img_i]['depth'][:,0]

    base_img=images[img_i]
    
    repj_imgs = torch.stack(repj_imgs)
    repj_poses = torch.stack(repj_poses)
    repj_intrs = torch.stack(repj_intrs)
    repj_depths = torch.stack(repj_depths)
    
    base_depends = base_img, base_depth, base_pose, base_intr
    repj_depends = repj_imgs, repj_depths, repj_poses, repj_intrs
    
    return base_coords, base_depends, repj_depends