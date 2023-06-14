import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def init_devices(args, rank, world_size):
    if args.distributed:
        _DEVICE = rank 
        # set_seed(seed)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(12358)
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        print(f"{rank + 1}/{world_size} process initialized.")
    else:
        _DEVICE = "cuda:0"
        world_size = torch.cuda.device_count()
    
    return _DEVICE

def dist_wrapper(args, model, rank, device):
    if args.distributed:
        model = DDP(model, device_ids=[rank], output_device=rank,
                    find_unused_parameters=False)
    else:
        gpus = [i for i in range(torch.cuda.device_count())]  
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
        model = nn.DataParallel(model,device_ids=gpus, output_device=gpus[0])  
        device = torch.cuda.current_device()
    return model, device