from model.loss import edge_aware_loss_v2
import torch
import torch.nn as nn

class RgbLoss(nn.Module):
    def __init__(self, args) -> None:
        super(RgbLoss, self).__init__()
        self.loss_fn = lambda x, y : torch.mean((x - y) ** 2)

    def forward(self, pred, tgt):
        return self.loss_fn(pred, tgt)

class SemanticLoss(nn.Module):
    def __init__(self, args) -> None:
        super(SemanticLoss, self).__init__()
        self.weight = args.semantic_lambda
        if args.semantic_loss_type == 'CE':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            print("Not Implemented Loss Type!")
            raise NotImplementedError

    def forward(self, pred, tgt):
        return self.loss_fn(pred, tgt) * self.weight

class DepthLoss(nn.Module):
    def __init__(self, args) -> None:
        super(DepthLoss, self).__init__()
        self.c_weight = args.coarse_depth_mult
        if(args.disparity_depth):
            self.loss_fn = lambda x, y: torch.abs(1/x - 1/y)
        else:
            self.loss_fn = lambda x, y: torch.abs(x - y)
        
    def forward(self, pred, pred_c, tgt):
        depth_loss = self.loss_fn(pred, tgt) + self.c_weight * self.loss_fn(pred_c, tgt)
        return depth_loss

class SmoothLoss(nn.Module):
    def __init__(self, args) -> None:
        super(SmoothLoss, self).__init__()
        self.use_skymask = args.skymask
        self.N_patch = args.N_patch
        self.patch_sz = args.patch_sz
        self.weight = args.smooth_lambda

    def forward(self, image, skymask, sel_coords_smooth, pred_distance_smooth):
        target_dep_rgb = torch.tensor(image[sel_coords_smooth[:, 0], sel_coords_smooth[:, 1]])
        sky = None
        if self.use_skymask:
            sky = skymask[sel_coords_smooth[:, 0], sel_coords_smooth[:, 1]]
            sky = sky.reshape(self.N_patch, self.patch_sz, self.patch_sz, -1)
        ori_rgb = target_dep_rgb.view(self.N_patch, self.patch_sz, self.patch_sz, -1)
        ori_disp = (1/torch.clamp(pred_distance_smooth, min=1e-5))
        ori_disp = ori_disp.view(self.N_patch, self.patch_sz, self.patch_sz, -1)
        smooth_loss = edge_aware_loss_v2(ori_rgb, ori_disp, sky) * self.weight
        return smooth_loss

class ProposalLoss(nn.Module):
    def __init__(self, args) -> None:
        super(ProposalLoss, self).__init__()
        self.weight = args.proposal_lambda

    def forward(self, s_vals_f, weights_f, s_vals_c, weights_c):
        s_vals_f = s_vals_f.detach()
        weights_f = weights_f.detach()
        inds=torch.searchsorted(s_vals_c,s_vals_f,right=True)
        W_c=torch.cumsum(weights_c,dim=1);
        left=torch.gather(W_c,1,torch.clamp(inds[:,:-1]-1,min=0.).long());
        right=torch.gather(W_c,1,torch.clamp(inds[:,1:]-1,max=weights_f.shape[1]-1))
        bound = right - left
        proposal_loss = torch.clamp(weights_f-bound,min=0)**2/(weights_f+1e-8)
        proposal_loss = proposal_loss.sum(axis=1).mean() * self.weight
        return proposal_loss 