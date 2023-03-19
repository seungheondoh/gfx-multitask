import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from mar.datasets.msd import CALS_Dataset
from mar.modules.audio_rep import TFRep
from mar.modules.tokenizer import ResFrontEnd, SpecPatchEmbed
from mar.classification.model import MusicTaggingTransformer
from mar.utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--framework', type=str, default="tagging") # or transcription
parser.add_argument('--data_dir', type=str, default="/music-semantic-dataset/dataset")
parser.add_argument('--arch', default='resnet50')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:12312', type=str,
                    help='url used to set up distributed training') # env://
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=50, type=int)
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# train detail
parser.add_argument("--duration", default=9.91, type=int)
parser.add_argument("--sr", default=16000, type=int)
parser.add_argument("--num_chunks", default=3, type=int)
parser.add_argument("--mel_dim", default=128, type=int)
parser.add_argument("--n_fft", default=1024, type=int)
parser.add_argument("--win_length", default=1024, type=int)
parser.add_argument("--tokenizer", default="cnn", type=str)
parser.add_argument("--mix_type", default="ft", type=str)
parser.add_argument("--spec_type", default="mel", type=str)
parser.add_argument("--cos", default=True, type=bool)
parser.add_argument("--attention_nheads", default=8, type=int)
parser.add_argument("--attention_nlayers", default=4, type=int)
parser.add_argument("--attention_ndim", default=256, type=int)



def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    audio_rep = TFRep(
                sample_rate= args.sr,
                f_min=0,
                f_max= int(args.sr / 2),
                n_fft = args.n_fft,
                win_length = args.win_length,
                hop_length = int(0.01 * args.sr),
                n_mels = args.mel_dim
            )
    if args.tokenizer == 'cnn':
        tokenizer = ResFrontEnd(
            input_size=(args.mel_dim, int(100 * args.duration) + 1),
            conv_ndim=128, 
            attention_ndim=args.attention_ndim,
            mix_type= args.mix_type
        )
    elif args.tokenizer == 'patch_proj':
        tokenizer = SpecPatchEmbed(
            f_size=128, t_size=int(100 * args.duration) + 1, p_w=16, p_h=16, in_chans=1, embed_dim=args.attention_ndim
        )
    model = MusicTaggingTransformer(
        audio_representation = audio_rep,
        tokenizer = tokenizer,
        spec_type = args.spec_type,
        attention_nheads = args.attention_nheads,
        attention_nlayers= args.attention_nlayers,
        attention_ndim= args.attention_ndim
    )
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    loss_fn = nn.BCELoss().to(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    earlystopping_callback = EarlyStopping()
    cudnn.benchmark = True

    train_dataset = CALS_Dataset(
        args.data_dir, "TRAIN", args.sr, args.duration, args.num_chunks, False
    )
    val_dataset = CALS_Dataset(
        args.data_dir, "VALID", args.sr, args.duration, args.num_chunks, False
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size//args.num_chunks, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)

    save_dir = f"exp/pretrain/{args.tokenizer}_{args.mix_type}/{args.spec_type}_{args.attention_nlayers}_{args.attention_ndim}_{args.attention_nheads}"
    logger = Logger(save_dir)
    save_hparams(args, save_dir)

    best_val_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, loss_fn, optimizer, epoch, logger, args)
        val_loss = validate(val_loader, model, loss_fn, epoch, args)
        logger.log_val_loss(val_loss, epoch)
        # save model
        if val_loss < best_val_loss:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, f'{save_dir}/best.pth')
            best_val_loss = val_loss

        # earlystopping_callback(val_loss, best_val_loss)
        # if earlystopping_callback.early_stop:
        #     print("We are at epoch:", epoch)
        #     break

def train(train_loader, model, loss_fn, optimizer, epoch, logger, args):
    train_losses = AverageMeter('Train Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),[train_losses],prefix="Epoch: [{}]".format(epoch))
    iters_per_epoch = len(train_loader)
    model.train()
    for data_iter_step, (x, y) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, data_iter_step / iters_per_epoch + epoch, args)
        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
            y = y.cuda(args.gpu, non_blocking=True)
        # compute output
        predict = model(x)
        loss = loss_fn(predict, y)
        train_losses.step(loss.item(), x.size(0))
        logger.log_train_loss(loss, epoch * iters_per_epoch + data_iter_step)
        logger.log_learning_rate(lr, epoch * iters_per_epoch + data_iter_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if data_iter_step % args.print_freq == 0:
            progress.display(data_iter_step)

def validate(val_loader, model, loss_fn, epoch, args):
    losses_val = AverageMeter('Valid Loss', ':.4e')
    progress_val = ProgressMeter(len(val_loader),[losses_val],prefix="Epoch: [{}]".format(epoch))
    model.eval()
    epoch_end_loss = []
    for data_iter_step, (x, y) in enumerate(val_loader):
        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
            y = y.cuda(args.gpu, non_blocking=True)
        B, C, T = x.size()
        with torch.no_grad():
            predict = model(x.view(-1, T)) # flatten batch
        avg_predict = predict.view(B, C, -1).mean(dim=1)
        loss = loss_fn(avg_predict, y)
        epoch_end_loss.append(loss.detach().cpu())
        losses_val.step(loss.item(), x.size(0))
        if data_iter_step % args.print_freq == 0:
            progress_val.display(data_iter_step)
    val_loss = torch.stack(epoch_end_loss).mean(0, False)
    return val_loss

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    main()

    