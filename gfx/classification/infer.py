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
from gfx.preprocessor.audio_utils import load_audio, STR_CH_FIRST
from gfx.modules.audio_rep import TFRep
from gfx.classification.model import ResNet

parser = argparse.ArgumentParser(description='PyTorch MSD Training')
parser.add_argument('--arch', default='resnet50')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=50, type=int)
parser.add_argument('--file_path', default="../../demo/01-d09-m03-t00-r00.mp3", type=str)
parser.add_argument("--duration", default=3, type=int)
parser.add_argument("--sr", default=22050, type=int)
parser.add_argument("--num_chunks", default=8, type=int)
parser.add_argument("--mel_dim", default=128, type=int)
parser.add_argument("--n_fft", default=1024, type=int)
parser.add_argument("--win_length", default=1024, type=int)
args = parser.parse_args()

LABELS = [
    "d01-clean",
    "d02-clean funk",
    "d03-jazz",
    "d04-blues",
    "d05-blues rock",
    "d06-funk rock",
    "d07-britpop",
    "d08-modern rock",
    "d09-hard rock",
    "d10-pop rock",
    "d11-alterantive rock",
    "d12-pop metal",
    "d13-heavy metal1",
    "d14-heavy metal2",
    "m00-none",
    "m01-chorus1",
    "m02-chorus2",
    "m03-chorus3",
    "m04-phaser1",
    "m05-phaser2",
    "m06-phaser3",
    "m07-flanger1",
    "m08-flanger2"
]
example_list = [
    "01-d09-m03-t00-r00.mp3", "04-d06-m00-t00-r00.mp3", "04-d14-m06-t00-r00.mp3"
]

def audio_preprocessor(audio):
    hop = (audio.shape[-1] - args.duration * args.sr) // 16
    audio_npy = np.stack([np.array(audio[i * hop : i * hop + args.duration * args.sr]) for i in range(16)]).astype('float32')
    audio_tensor = torch.from_numpy(audio_npy)
    return audio_tensor

def predict():
    save_dir = "./exp"
    src, _ = load_audio(
        path= args.file_path,
        ch_format= STR_CH_FIRST,
        sample_rate= args.sr,
        downmix_to_mono= True)
    audio_tensor = audio_preprocessor(src)
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
    pretrained_object = torch.load(f'{save_dir}/best.pth', map_location='cpu')
    state_dict = pretrained_object['state_dict']
    model.load_state_dict(state_dict)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    cudnn.benchmark = True

    model.eval()
    with torch.no_grad():
        pred_probs = model(audio_tensor.to(args.gpu))
    pred_probs = pred_probs.detach().cpu().mean(0,False)
    pred_labels_and_probs = {LABELS[i]: float(pred_probs[i]) for i in range(len(LABELS))}    
    print("===========")
    print(args.file_path)
    print (sorted([(v,k) for k, v in pred_labels_and_probs.items()], reverse=True))
    print("===========")

if __name__ == '__main__':
    predict()

    