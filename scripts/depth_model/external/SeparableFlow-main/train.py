from __future__ import print_function
import argparse
from math import log10
import sys
sys.path.append('core')
import shutil
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from sepflow import SepFlow
import evaluate
import datasets
from torch.utils.tensorboard import SummaryWriter
from utils.utils import InputPadder, forward_interpolate

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# Training settings
parser = argparse.ArgumentParser(description='PyTorch SepFlow Example')
parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--weights', type=str, default='', help="weights from saved model")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2048, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--manual_seed', type=int, default=1234, help='random seed to use. Default=123')
parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')
parser.add_argument('--data_path', type=str, default='/export/work/feihu/flow/SceneFlow/', help="data root")
parser.add_argument('--save_path', type=str, default='./checkpoints/', help="location to save models")
parser.add_argument('--gpu',  default='0,1,2,3,4,5,6,7', type=str, help="gpu idxs")
parser.add_argument('--workers', type=int, default=16, help="workers")
parser.add_argument('--world_size', type=int, default=1, help="world_size")
parser.add_argument('--rank', type=int, default=0, help="rank")
parser.add_argument('--dist_backend', type=str, default="nccl", help="dist_backend")
parser.add_argument('--dist_url', type=str, default="tcp://127.0.0.1:6789", help="dist_url")
parser.add_argument('--distributed', type=int, default=0, help="distribute")
parser.add_argument('--sync_bn', type=int, default=0, help="sync bn")
parser.add_argument('--multiprocessing_distributed', type=int, default=0, help="multiprocess")
parser.add_argument('--freeze_bn', type=int, default=0, help="freeze bn")
parser.add_argument('--start_epoch', type=int, default=0, help="start epoch")
parser.add_argument('--stage', type=str, default='chairs', help="training stage: 1) things 2) chairs 3) kitti 4) mixed.")
parser.add_argument('--validation', type=str, nargs='+')
parser.add_argument('--num_steps', type=int, default=100000)
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--iters', type=int, default=12)
parser.add_argument('--wdecay', type=float, default=.00005)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
parser.add_argument('--add_noise', action='store_true')
parser.add_argument('--small', action='store_true', help='use small model')
#parser.add_argument('--smoothl1', action='store_true', help='use smooth l1 loss')

MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 2500

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    weights = [0.1, 0.3, 0.5]
    base = weights[2] - gamma ** (n_predictions - 3)
    for i in range(n_predictions - 3):
        weights.append( base + gamma**(n_predictions - i - 4) )

    for i in range(n_predictions):
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += weights[i] * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    loss_value = flow_loss.detach()
    rate0 = (epe > 1).float().mean()
    rate1 = (epe > 3).float().mean()
    error3 = epe.mean()
    epe = torch.sum((flow_preds[1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    error1 = epe.mean()
    epe = torch.sum((flow_preds[0] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    error0 = epe.mean()
    epe = torch.sum((flow_preds[2] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    error2 = epe.mean()
    
    if args.multiprocessing_distributed:
        count = flow_gt.new_tensor([1], dtype=torch.long)
        dist.all_reduce(loss_value), dist.all_reduce(error3), dist.all_reduce(error0), dist.all_reduce(error1), dist.all_reduce(error2), dist.all_reduce(count)
        dist.all_reduce(rate0), dist.all_reduce(rate1)
        n = count.item()
        loss_value, error0, error1, error2, error3 = loss_value / n, error0 / n, error1 / n, error2 / n, error3 / n
        rate1, rate0 = rate1 / n, rate0 / n

    metrics = {
        'epe0': error0.item(),
        'epe1': error1.item(),
        'epe2': error2.item(),
        'epe3': error3.item(),
        '1px': rate0.item(),
        '3px': rate1.item(),
        'loss': loss_value.item()
    }
    return flow_loss, metrics

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port
def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.gpu = (args.gpu).split(',')
    torch.backends.cudnn.benchmark = True
   # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu.split(','))
    #args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if args.manual_seed is not None:
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = True
        cudnn.deterministic = True
    args.ngpus_per_node = len(args.gpu)
    if len(args.gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        main_worker(args.gpu, args.ngpus_per_node, args)
    else:
        args.sync_bn = True
        args.distributed = True
        args.multiprocessing_distributed = True
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        #print(args)
        #quit()
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    modules_ori = [model.cnet, model.fnet, model.update_block, model.guidance]
    modules_new = [model.cost_agg1, model.cost_agg2]
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.lr * 2.5))
    #optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    optimizer = optim.AdamW(params_list, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)


    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)
def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    model = SepFlow(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batchSize = int(args.batchSize / args.ngpus_per_node)
        args.testBatchSize = int(args.testBatchSize / args.ngpus_per_node)
        args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model).cuda()

    #scheduler = None
    logger = Logger(model, scheduler)


    if args.weights:
        if os.path.isfile(args.weights):
            checkpoint = torch.load(args.weights, map_location=lambda storage, loc: storage.cuda())
            msg=model.load_state_dict(checkpoint['state_dict'], strict=False)
            if main_process():
                print("=> loaded checkpoint '{}'".format(args.weights))
                print(msg)
                sys.stdout.flush()
        else:
            if main_process():
                print("=> no checkpoint found at '{}'".format(args.weights))
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            msg=model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            args.start_epoch = checkpoint['epoch'] + 1
            if main_process():
                print("=> resume checkpoint '{}'".format(args.resume))
                print(msg)
                sys.stdout.flush()
        else:
            if main_process():
                print("=> no checkpoint found at '{}'".format(args.resume))

    train_set = datasets.fetch_dataloader(args)
    val_set = datasets.KITTI(split='training')
    val_set3 = datasets.FlyingChairs(split='validation')
    val_set2_2 = datasets.MpiSintel(split='training', dstype='final')
    val_set2_1 = datasets.MpiSintel(split='training', dstype='clean')
    sys.stdout.flush()
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        val_sampler2_2 = torch.utils.data.distributed.DistributedSampler(val_set2_2)
        val_sampler2_1 = torch.utils.data.distributed.DistributedSampler(val_set2_1)
        val_sampler3 = torch.utils.data.distributed.DistributedSampler(val_set3)
    else:
        train_sampler = None
        val_sampler = None
        val_sampler2_1 = None
        val_sampler2_2 = None
        val_sampler3 = None
    training_data_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batchSize, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=args.testBatchSize, shuffle=False, num_workers=args.workers//2, pin_memory=True, sampler=val_sampler)
    val_data_loader2_2 = torch.utils.data.DataLoader(val_set2_2, batch_size=args.testBatchSize, shuffle=False, num_workers=args.workers//2, pin_memory=True, sampler=val_sampler2_2)
    val_data_loader2_1 = torch.utils.data.DataLoader(val_set2_1, batch_size=args.testBatchSize, shuffle=False, num_workers=args.workers//2, pin_memory=True, sampler=val_sampler2_1)
    val_data_loader3 = torch.utils.data.DataLoader(val_set3, batch_size=args.testBatchSize, shuffle=False, num_workers=args.workers//2, pin_memory=True, sampler=val_sampler3)

    error = 100
    args.nEpochs = args.num_steps // len(training_data_loader) + 1

    for epoch in range(args.start_epoch, args.nEpochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(training_data_loader, model, optimizer, scheduler, logger, epoch)
        if main_process() and epoch > args.nEpochs - 3:
            save_checkpoint(args.save_path, epoch,{
                    'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer' : optimizer.state_dict(),
                     'scheduler' : scheduler.state_dict(),
                 }, False)
        
        if args.stage == 'chairs':
            loss = val(val_data_loader3, model, split='chairs')
        elif args.stage == 'sintel' or args.stage == 'things':
            loss_tmp = val(val_data_loader2_1, model, split='sintel', iters=32)
            loss_tmp = val(val_data_loader2_2, model, split='sintel', iters=32)
            loss_tmp = val(val_data_loader, model, split='kitti')
        elif args.stage == 'kitti':
            loss_tmp = val(val_data_loader, model, split='kitti')

    if main_process():
        save_checkpoint(args.save_path, args.nEpochs,{
                'state_dict': model.state_dict()
            }, True)




def train(training_data_loader, model, optimizer, scheduler, logger, epoch):
    valid_iteration = 0
    model.train()
    if args.freeze_bn:
        model.module.freeze_bn()
        if main_process():
            print("Epoch " + str(epoch) + ": freezing bn...")
            sys.stdout.flush()
    for iteration, batch in enumerate(training_data_loader):
        input1, input2, target, valid = Variable(batch[0], requires_grad=True), Variable(batch[1], requires_grad=True), Variable(batch[2], requires_grad=False), Variable(batch[3], requires_grad=False)
        input1 = input1.cuda(non_blocking=True)
        input2 = input2.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        valid = valid.cuda(non_blocking=True)
        if len(valid.shape) > 3:
            valid = valid.squeeze(1)
        if valid.sum() > 0:
            optimizer.zero_grad()
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                input1 = (input1 + stdv * torch.randn(*input1.shape).cuda()).clamp(0.0, 255.0)
                input2 = (input2 + stdv * torch.randn(*input2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(input1, input2, iters=args.iters)            
            loss, metrics = sequence_loss(flow_predictions, target, valid)

            loss.backward()
            optimizer.step()
            scheduler.step()
            adjust_learning_rate(optimizer, scheduler)
            if  scheduler.get_last_lr()[0] < 0.0000002:
                return

            
            valid_iteration += 1

            if main_process():
                logger.push(metrics)
         #       print(metrics)
                if valid_iteration % 10000 == 0: 
                    save_checkpoint(args.save_path, epoch,{
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'scheduler' : scheduler.state_dict(),
                        }, False)

            sys.stdout.flush()

def val(testing_data_loader, model, split='sintel', iters=24):
    epoch_error = 0
    epoch_error_rate0 = 0
    epoch_error_rate1 = 0
    valid_iteration = 0
    model.eval()
    for iteration, batch in enumerate(testing_data_loader):
        input1, input2, target, valid = Variable(batch[0],requires_grad=False), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False), Variable(batch[3], requires_grad=False)
        input1 = input1.cuda(non_blocking=True)
        input2 = input2.cuda(non_blocking=True)
        padder = InputPadder(input1.shape, mode=split)
        input1, input2 = padder.pad(input1, input2)
        target = target.cuda(non_blocking=True)
        valid = valid.cuda(non_blocking=True)
        mag = torch.sum(target**2, dim=1, keepdim=False).sqrt()
        if len(valid.shape) > 3:
            valid = valid.squeeze(1)
        valid = (valid >= 0.001) #& (mag < MAX_FLOW)
        if valid.sum()>0:
            with torch.no_grad():
                _, flow = model(input1,input2, iters=iters)
                flow = padder.unpad(flow)
                epe = torch.sum((flow - target)**2, dim=1).sqrt()
                epe = epe.view(-1)[valid.view(-1)]
                rate0 = (epe > 1).float().mean()
                if split == 'kitti':
                    rate1 = ((epe > 3.0) & ((epe/mag.view(-1)[valid.view(-1)]) > 0.05)).float().mean()
                else:
                    rate1 = (epe > 3.0).float().mean()
                error = epe.mean()
                valid_iteration += 1
            if args.multiprocessing_distributed:
                count = target.new_tensor([1], dtype=torch.long)
                dist.all_reduce(error)
                dist.all_reduce(rate0)
                dist.all_reduce(rate1)
                dist.all_reduce(count)
                n = count.item()
                error /= n
                rate0 /= n
                rate1 /= n
                epoch_error += error.item()
                epoch_error_rate0 += rate0.item()
                epoch_error_rate1 += rate1.item()

            if main_process() and (valid_iteration % 1000 == 0):
                print("===> Test({}/{}): Error: ({:.4f} {:.4f} {:.4f})".format(iteration, len(testing_data_loader), error.item(), rate0.item(), rate1.item()))
            sys.stdout.flush()

    if main_process():
        print("===> Test: Avg. Error: ({:.4f} {:.4f} {:.4f})".format(epoch_error/valid_iteration, epoch_error_rate0/valid_iteration, epoch_error_rate1/valid_iteration))

    return epoch_error/valid_iteration

def save_checkpoint(save_path, epoch,state, is_best):
    filename = save_path + "_epoch_{}.pth".format(epoch)
    if is_best:
        filename = save_path + ".pth"
    torch.save(state, filename)
    print("Checkpoint saved to {}".format(filename))

def adjust_learning_rate(optimizer, scheduler):
    lr = scheduler.get_last_lr()[0]
    nums = len(optimizer.param_groups)
    for index in range(0, nums-2):
        optimizer.param_groups[index]['lr'] = lr
    for index in range(nums-2, nums):
        optimizer.param_groups[index]['lr'] = lr * 2.5

if __name__ == '__main__':
    main()
