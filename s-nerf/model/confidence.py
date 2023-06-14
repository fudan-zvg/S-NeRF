import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from model.loss import get_reproj_conf

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().to(device)
        self.device = device
        self.criterion = nn.L1Loss(reduction='none')
        self.weights = [1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):            
        x = x.permute(2,0,1).float()
        y = y.permute(2,0,1).float()
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        H,W=x.shape[1:]
        loss = torch.zeros((H,W)).to(self.device)
        for i in range(len(x_vgg)-1):
            feat_x, feat_y = x_vgg[i], y_vgg[i].detach()
            if(i > 0):
                feat_x = F.upsample(x_vgg[i].unsqueeze(0), mode='bilinear', size=(H,W), align_corners=True)
                feat_y = F.upsample(y_vgg[i].unsqueeze(0), mode='bilinear', size=(H,W), align_corners=True)
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
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)          
        out = [h_relu1, h_relu2, h_relu3, h_relu4]
        return out

class Confidence(nn.Module):
    def __init__(self, modes, device,images_num, vgg_loss=True,tau=0.2):
        super(Confidence, self).__init__()        
        self.modes = modes
        self.lambdas = nn.Parameter(torch.zeros((len(modes),images_num)),requires_grad=True)
        self.vgg_loss = None
        if vgg_loss:
            self.vgg_loss = VGGLoss(device)
        self.tau=tau
    
    '''
    Save the confidence to the given path
    '''
    def precompute_conf_map(self, args,cam_index,images, depth_gts,poses,intrinsics,i_train,flow=None):
        conf_maps=[]
        for i in range(len(i_train)):
            img_i=i_train[i]
            base_depends, repj_depends,flow_depends=select_conf_depends(args, img_i, cam_index, images, depth_gts, poses, intrinsics, i_train, pose_param_net=None,flow=flow)
            import pdb; pdb.set_trace()
            conf_map = self.forward(base_depends, repj_depends,img_i,flow_depends,ret_dict=True)
            conf_maps.append(conf_map)
        return conf_maps
        
    '''
    Project base idx images to others to calculate confidence
    '''
    def forward(self, base_depends, repj_depends, img_i, flow_depends=None, ret_dict=False):
        base_img = base_depends[0]
        proj_coord = torch.nonzero(torch.ones((base_img.shape[0], base_img.shape[1])))
        proj_coord = torch.flip(proj_coord, [1])
        conf_maps = get_reproj_conf(self.modes,
                        proj_coord, 
                        base_depends, 
                        repj_depends,
                        flow_depends,
                        self.vgg_loss,
                        self.tau)

        H, W = base_img.shape[:2]
        final_conf_map = torch.zeros((H,W))
        weights=F.sigmoid(self.lambdas[:,img_i])

        for idx, mode in enumerate(self.modes):
            final_conf_map += weights[idx] * conf_maps[mode].reshape(H,W)
        final_conf_map /= weights.sum()
        if not ret_dict:
            return final_conf_map
        else:
            return conf_maps

        
def select_conf_depends(args, img_i, cam_index, images, depth_gts, poses, intrinsics, i_train, pose_param_net=None, flow=None,flow_interval=1):
    repj_imgs = []
    repj_poses = []
    repj_intrs = []
    repj_depths = []

    base_img = images[img_i]
    base_depth = depth_gts[img_i]
    base_pose = pose_param_net(img_i) if pose_param_net is not None else poses[img_i]
    base_intr = intrinsics[img_i]
    img_idx = np.where(i_train == img_i)[0].item()
    # cam_index shows the img_idx with the cam_idx
    sel_ids=[]
    for k in range(1,args.conf_num+1):
        # confirm the index is in the [0,images.shape[0]-1] and below to the same camera
        if img_idx+k < len(i_train):
            if cam_index[i_train[img_idx]]==cam_index[i_train[img_idx+k]]:
                sel_ids.append(i_train[img_idx+k])
        if img_idx-k >=0:
            if cam_index[i_train[img_idx]]==cam_index[i_train[img_idx-k]]:
                sel_ids.append(i_train[img_idx-k])

    for sel_id in sel_ids:
        repj_imgs.append(images[sel_id])
        repj_poses.append(pose_param_net(sel_id) if pose_param_net is not None else poses[sel_id])
        repj_depths.append(depth_gts[sel_id])
        repj_intrs.append(intrinsics[sel_id])
    flow_depends=None        
    repj_flows=[];flow_imgs=[];flow_intrs=[];flow_depths=[];flow_poses=[]
    if flow is not None:
        # img_idx = np.where(i_train == img_i)[0].item()
        if img_i+flow_interval in i_train:# this is for next_flows
            if cam_index[img_i]==cam_index[img_i+flow_interval]:
                repj_flows.append(flow[img_i+flow_interval][0])
                flow_imgs.append(images[img_i+flow_interval])
                flow_intrs.append(intrinsics[img_i+flow_interval])
                flow_poses.append(poses[img_i+flow_interval])
                flow_depths.append(depth_gts[img_i+flow_interval])
        if img_i-flow_interval in i_train:# this is for prev_flows
            if cam_index[img_i]==cam_index[img_i-flow_interval]:
                repj_flows.append(flow[img_i-flow_interval][1])
                flow_imgs.append(images[img_i-flow_interval])
                flow_intrs.append(intrinsics[img_i-flow_interval])
                flow_poses.append(poses[img_i-flow_interval])
                flow_depths.append(depth_gts[img_i-flow_interval])
        
        flow_depends=torch.tensor(flow_imgs),torch.tensor(flow_depths),flow_poses,flow_intrs,torch.tensor(repj_flows)
   

    repj_imgs = torch.tensor(np.stack(repj_imgs))
    repj_poses = torch.stack(repj_poses)
    repj_intrs = torch.stack(repj_intrs)
    repj_depths = torch.tensor(np.stack(repj_depths))
    
    if(args.flow):
        assert args.conf_num == 1
    
    base_depends = torch.tensor(base_img), torch.tensor(base_depth), base_pose, base_intr
    repj_depends = repj_imgs, repj_depths, repj_poses, repj_intrs


    return base_depends, repj_depends, flow_depends

def build_confidence_model(args, N, device):
    mode=[]
    if not args.no_reproj:
        mode+=['rgb','ssim']
    if not args.no_geometry:
        mode+=['depth']
    if args.vgg_loss:
        mode+=['vgg']
    Conf = Confidence(mode,
        device=device, images_num=N,
        vgg_loss=args.vgg_loss, tau=args.tau)
    optimizer_conf=torch.optim.Adam([
        {'params':[p for p in Conf.parameters() if p.requires_grad], 'lr':1e-3}],
    )
    return Conf, optimizer_conf

def calc_final_confidence(args, target_mask, confidence_depends):
    Conf, sel_coords, img_i, cam_index, images, depth_gts, poses, intrinsics, i_train, pose_param_net, all_conf_maps, skymask = confidence_depends
    if not args.precompute_conf:
        base_depends, repj_depends, flow_depends = select_conf_depends(args, img_i, cam_index, images, depth_gts, poses, intrinsics, i_train, pose_param_net)
        conf_map = Conf(base_depends, repj_depends, img_i)
    else:
        conf_maps = all_conf_maps[np.where(i_train==img_i)[0].item()]
        H, W = images[img_i].shape[:2]
        final_conf_map = torch.zeros((H,W))
        weights=F.sigmoid(Conf.lambdas[:,img_i])
        for idx, mode in enumerate(Conf.modes):
            final_conf_map += weights[idx] * conf_maps[mode].reshape(H,W)
        final_conf_map/=weights.sum()
        conf_map=final_conf_map

    if args.skymask:
        conf_map[skymask[img_i]] = 1

    conf_arr = conf_map[sel_coords[:,0],sel_coords[:,1]]
    confidence = conf_arr.unsqueeze(-1)[target_mask]
    return confidence

def calc_depth_loss(args, backcam_mask, target_depth, pred_distance, pred_distance_coarse, depth_loss_fn, confidence_depends):
    img_i, cam_index = confidence_depends[2], confidence_depends[3]
    target_mask = target_depth!=0
    target_depth = target_depth.unsqueeze(-1)
    pred_distance = pred_distance.unsqueeze(-1)
    pred_distance_coarse = pred_distance_coarse.unsqueeze(-1)

    depth_loss = depth_loss_fn(pred_distance[target_mask], pred_distance_coarse[target_mask], target_depth[target_mask])

    if(args.depth_conf): # this time is the single image training mode
        confidence = calc_final_confidence(args, target_mask, confidence_depends)
        depth_loss *= confidence
    
    if args.backcam and cam_index[img_i] == 0:
        depth_loss = depth_loss[backcam_mask[target_mask]]

    return depth_loss.mean()