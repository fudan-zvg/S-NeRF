import torch
import torch.nn as nn
import numpy as np
import torch


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def vec2skew(v):
    """
    :param v:  (N, 3, ) torch tensor
    :return:   (N, 3, 3)
    """
    N = v.size(0)
    zero = torch.zeros([N,1], dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[:,2:3],   v[:,1:2]], dim=1)  # (N, 3, 1)
    skew_v1 = torch.cat([ v[:,2:3],   zero,    -v[:,0:1]], dim=1)
    skew_v2 = torch.cat([-v[:,1:2],   v[:,0:1],   zero], dim=1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)  # (N, 3, 3)
    return skew_v  # (N, 3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (N, 3, ) axis-angle, torch tensor
    :return:  (N, 3, 3)
    """
    skew_r = vec2skew(r)  # (N, 3, 3)
    norm_r = r.norm(dim=-1)[:,None, None] + 1e-15 # (N, 1, 1)
    eye = torch.eye(3, dtype=torch.float32, device=r.device)[None]
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R


def make_c2w(r, t):
    """
    :param r:  (N, 3, ) axis-angle             torch tensor
    :param t:  (N, 3, ) translation vector     torch tensor
    :return:   (N, 4, 4)
    """
    R = Exp(r)  # (N, 3, 3)
    c2w = torch.cat([R, t.unsqueeze(2)], dim=2)  # (N, 3, 4)
    c2w = convert3x4_4x4(c2w)  # (N, 4, 4)
    return c2w


class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R=True, learn_t=True, init_c2w=None, t_ratio=1):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4)
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        self.t_ratio = t_ratio

        if init_c2w is not None:
            init_c2w = torch.tensor(init_c2w)
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id, transform_only=False, one_img=False):
        if one_img:
            r = self.r[cam_id[0]]  # (3, ) axis-angle
            t = self.t[cam_id[0]]*self.t_ratio  # (3, )
            c2w = make_c2w(r, t)  # (4, 4)
            return c2w[None]

        # c2ws = []
        # for i in range(self.num_cams):
        #     r = self.r[i]  # (3, ) axis-angle
        #     t = self.t[i]  # (3, )
        #     c2w = make_c2w(r, t)  # (4, 4)
        #     c2ws.append(c2w)
        c2ws = make_c2w(self.r, self.t*self.t_ratio)
        c2w = c2ws[cam_id]

        if transform_only:
            return c2w
        else:
            # learn a delta pose between init pose and target pose, if a init pose is provided
            if self.init_c2w is not None:
                c2w = c2w @ self.init_c2w[cam_id]
            return c2w

if __name__=='__main__':
    posenet = LearnPose(5, True, True)
    cam_id = torch.tensor([1,2,3,4,0])
    print(posenet(cam_id))