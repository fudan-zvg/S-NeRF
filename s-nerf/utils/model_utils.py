import torch
import math
import os
from model.models import make_mipnerf
from model.poses import LearnPose
from torch.utils.tensorboard import SummaryWriter
from utils.device_utils import set_seed, init_devices, dist_wrapper

def learning_rate_decay(step, lr_init=5e-4, lr_final=5e-6, max_steps=1000000,
                        lr_delay_steps=2500, lr_delay_mult=0.01):
    step=torch.tensor(step)
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * torch.sin(
            0.5 * math.pi * torch.clamp(step / lr_delay_steps, 0, 1))
    else:
        delay_rate = 1.
    t = torch.clamp(step / max_steps, 0, 1)
    log_lerp = torch.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)
    return delay_rate * log_lerp


def build_main_model(args, rank, device):
    model = make_mipnerf(args, device=rank)
    model, device = dist_wrapper(args, model, rank, device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_parameters = sum([p.numel() for p in model_parameters])

    optimizer = torch.optim.Adam(params=[
        p for p in model.parameters() if p.requires_grad
    ], weight_decay=args.weight_decay_mult/model_parameters)

    return model, optimizer, device

def build_pose_ref_model(args, poses, _DEVICE):
    pose_param_net = LearnPose(num_cams=poses.shape[0], learn_R=True, learn_t=args.translation, init_c2w=poses)
    pose_param_net = pose_param_net.to(_DEVICE)
    optimizer_pose = torch.optim.Adam([
        {'params':[p for p in pose_param_net.parameters() if p.requires_grad], 'lr':1e-4}],
    )
    return pose_param_net, optimizer_pose

def resume_training(args, model, optimizer, pose_param_net, optimizer_pose, Conf):
    if args.resume:
        traindir=os.path.join(args.basedir, args.expname)
        files = [f for f in os.listdir(traindir) if f.endswith('tar')]
        ckpts=sorted(files,key= lambda x: (x.split('.'))[0])

        ckpt= torch.load(os.path.join(traindir,ckpts[-1]))
        model.load_state_dict(ckpt['model_param'])
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimzer'])
        if args.pose_refine:
            ckpt_pose=torch.load(os.path.join(traindir,'pose',ckpts[-1]))
            pose_param_net.load_state_dict(ckpt_pose['model_param'])
            optimizer_pose.load_state_dict(ckpt_pose['optimizer'])
        if args.depth_conf:
            Conf.load_state_dict(ckpt['confidence'])
    else:
        start = 0
    
    return start, model, optimizer, pose_param_net, optimizer_pose, Conf

def exp_dir_prep(args):
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    writer = SummaryWriter(os.path.join(basedir, expname))
    if args.pose_refine:
        os.makedirs(os.path.join(basedir, expname,'pose'), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    return writer