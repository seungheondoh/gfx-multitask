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

from mar.datasets.msd import CALS_Dataset
from mar.modules.audio_rep import TFRep
from mar.modules.tokenizer import ResFrontEnd, SpecPatchEmbed
from mar.models.tagging_model import MusicTaggingTransformer
from mar.utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams

TAGNAMES = [
    'rock','pop','indie','alternative','electronic','hip-hop','metal','jazz','punk',
    'folk','alternative rock','indie rock','dance','hard rock','00s','soul','hardcore',
    '80s','country','classic rock','punk rock','blues','chillout','experimental',
    'heavy metal','death metal','90s','reggae','progressive rock','ambient','acoustic',
    'beautiful','british','rnb','funk','metalcore','mellow','world','guitar','trance',
    'indie pop','christian','house','spanish','latin','psychedelic','electro','piano',
    '70s','progressive metal',
]

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--framework', type=str, default="tagging") # or transcription
parser.add_argument('--data_dir', type=str, default="/music-semantic-dataset/dataset")
parser.add_argument('--arch', default='resnet50')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N')
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
parser.add_argument('--gpu', default=3, type=int,
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
parser.add_argument("--num_chunks", default=8, type=int)
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
    main_worker(args)

def main_worker(args):
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
            input_size=(args.mel_dim, int(100 * args.duration)),
            conv_ndim=128, 
            attention_ndim=256,
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
    save_dir = f"exp/pretrain/{args.tokenizer}_{args.mix_type}/{args.spec_type}_{args.attention_nlayers}_{args.attention_ndim}_{args.attention_nheads}"
    print(save_dir)
    pretrained_object = torch.load(f'{save_dir}/best.pth', map_location='cpu')
    state_dict = pretrained_object['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict)


    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    cudnn.benchmark = True

    test_dataset = CALS_Dataset(
        args.data_dir, "TEST", args.sr, args.duration, args.num_chunks, False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)
    
    model.eval()
    predictions, groudturths = [], []
    for batch in tqdm(test_loader):
        x,y = batch
        if args.gpu is not None:
            x = x.cuda(args.gpu, non_blocking=True)
            y = y.cuda(args.gpu, non_blocking=True)
        B, C, T = x.size()
        with torch.no_grad():
            predict = model(x.view(-1, T)) # flatten batch
        avg_predict = predict.view(B, C, -1).mean(dim=1)
        predictions.append(avg_predict.detach().cpu())
        groudturths.append(y.detach().cpu())
    
    logits = torch.cat(predictions, dim=0)
    targets = torch.cat(groudturths, dim=0)
    roc_auc = metrics.roc_auc_score(targets, logits, average='macro')
    pr_auc = metrics.average_precision_score(targets, logits, average='macro')
    results = {
        'roc_auc' :roc_auc,
        'pr_auc': pr_auc
    }
    # tag wise score
    roc_aucs = metrics.roc_auc_score(targets, logits, average=None)
    pr_aucs = metrics.average_precision_score(targets, logits, average=None)
    tag_wise = {}
    for i in range(50):
        tag_wise[TAGNAMES[i]] = {
            "roc_auc":roc_aucs[i], 
            "pr_auc":pr_aucs[i]
    }
    results['tag_wise'] = tag_wise
    print(results)
    with open(os.path.join(save_dir, f"results.json"), mode="w") as io:
        json.dump(results, io, indent=4)
    

    

if __name__ == '__main__':
    main()

    