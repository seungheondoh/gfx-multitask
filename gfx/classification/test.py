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
from sklearn import metrics
from tqdm import tqdm
import json

from gfx.classification.dataset import AUGUST_Dataset
from gfx.modules.audio_rep import TFRep
from gfx.modules.tokenizer import ResFrontEnd, SpecPatchEmbed
from gfx.classification.model import ResNet
from gfx.utils.eval_utils import get_binary_decisions, save_cm

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--framework', type=str, default="tagging") # or transcription
parser.add_argument('--data_dir', type=str, default="../../dataset")
parser.add_argument('--arch', default='resnet50')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=50, type=int)

parser.add_argument("--duration", default=3, type=int)
parser.add_argument("--sr", default=22050, type=int)
parser.add_argument("--num_chunks", default=8, type=int)
parser.add_argument("--mel_dim", default=128, type=int)
parser.add_argument("--n_fft", default=1024, type=int)
parser.add_argument("--win_length", default=1024, type=int)
parser.add_argument("--tokenizer", default="cnn", type=str)
parser.add_argument("--mix_type", default="ft", type=str)
parser.add_argument("--spec_type", default="mel", type=str)
parser.add_argument("--cos", default=True, type=bool)

def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    random.seed(42)
    torch.manual_seed(42)
    cudnn.deterministic = True
    audio_rep = TFRep(
                sample_rate= args.sr,
                f_min=0,
                f_max= int(args.sr / 2),
                n_fft = args.n_fft,
                win_length = args.win_length,
                hop_length = int(0.01 * args.sr),
                n_mels = args.mel_dim
            )
    
    model = ResNet(
        audio_representation = audio_rep
    )

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    loss_fn = nn.BCELoss().to(args.gpu)
    cudnn.benchmark = True

    test_dataset = AUGUST_Dataset(
        args.data_dir, "TEST", args.sr, args.duration, args.num_chunks
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    save_dir = f"exp/"
    pretrained_object = torch.load(f'{save_dir}/best.pth', map_location='cpu')
    state_dict = pretrained_object['state_dict']
    model.load_state_dict(state_dict)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    cudnn.benchmark = True

    test_dataset = AUGUST_Dataset(
        args.data_dir, "TEST", args.sr, args.duration, args.num_chunks
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=None,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)
    
    model.eval()
    predictions, groudturths = [], []
    for batch in tqdm(test_loader):
        x = batch['audio']
        y = batch['binary']
        track_id = batch['track_id']
        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
            y = y.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            pred_probs = model(x.squeeze(0))
        pred_probs = pred_probs.mean(0,False)
        predictions.append(pred_probs.detach().cpu())
        groudturths.append(y.squeeze(0).detach().cpu())

    logits = torch.stack(predictions).numpy()
    targets = torch.stack(groudturths).numpy()
    roc_auc = metrics.roc_auc_score(targets, logits, average='macro')
    pr_auc = metrics.average_precision_score(targets, logits, average='macro')
    best_f1, bset_macro_f1, best_decisions, thresholds = get_binary_decisions(targets, logits, best_f1=True)
    sample_f1, macro_f1, decisions, _ = get_binary_decisions(targets, logits, best_f1=False)
    results = {
        'roc_auc' :roc_auc,
        'pr_auc': pr_auc,
        "best_f1": best_f1,
        "bset_macro_f1": bset_macro_f1,
        "sample_f1": sample_f1,
        "macro_f1": macro_f1
    }
    print(results)
    with open(os.path.join(save_dir, f"results.json"), mode="w") as io:
        json.dump(results, io, indent=4)
    save_cm(best_decisions, targets, list(test_dataset.fl.columns), os.path.join(save_dir, "best_cm.png"))
    save_cm(decisions, targets, list(test_dataset.fl.columns), os.path.join(save_dir, "cm.png"))
    

    

if __name__ == '__main__':
    main()

    