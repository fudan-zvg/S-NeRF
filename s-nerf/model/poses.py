import torch
import torch.nn as nn
from utils.lie_group_helper import make_c2w


class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4)
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        init_c2w = torch.tensor(init_c2w)
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id,transform_only=False):
        r = self.r[cam_id]  # (3, ) axis-angle
        
        t = self.t[cam_id]  # (3, )
        
        c2w = make_c2w(r, t)  # (4, 4)
       
        if transform_only:
            return c2w
        else:
            # learn a delta pose between init pose and target pose, if a init pose is provided
            if self.init_c2w is not None:
                c2w = c2w @ self.init_c2w[cam_id]
            return c2w