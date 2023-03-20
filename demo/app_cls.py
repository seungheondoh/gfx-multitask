import gradio as gr
from timeit import default_timer as timer
import gdown
import torch
import numpy as np
from gfx.modules.audio_rep import TFRep
from gfx.classification.model import ResNet
from gfx.preprocessor.constants import MUSIC_SAMPLE_RATE
from gfx.preprocessor.audio_utils import load_audio, STR_CH_FIRST
import random
import torch.backends.cudnn as cudnn

random.seed(42)
torch.manual_seed(42)
cudnn.deterministic = True
url = 'https://drive.google.com/uc?id=1Sa-K55HIvti2z3qpcTt7678Mkhz46czT'
output= 'best.pth'
# gdown.download(url, output, quiet=False)

device = "cpu"
AUDIO_LEN = 3
model = torch.load(output, map_location=torch.device('cpu'))
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
    hop = (len(audio) - AUDIO_LEN * MUSIC_SAMPLE_RATE) // 8
    audio_npy = np.stack([np.array(audio[i * hop : i * hop + AUDIO_LEN * MUSIC_SAMPLE_RATE]) for i in range(8)]).astype('float32')
    audio_tensor = torch.from_numpy(audio_npy)
    return audio_tensor

def predict(audio_path):
    start_time = timer()
    src, _ = load_audio(
        path= audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)

    if len(src) > int(AUDIO_LEN * MUSIC_SAMPLE_RATE):
        src = src[:int(AUDIO_LEN * MUSIC_SAMPLE_RATE)]
    else:
        print(f"input length {len(wav)} too small!, need over {int(AUDIO_LEN * MUSIC_SAMPLE_RATE)}")
        return
    audio_tensor = audio_preprocessor(src)
    audio_rep = TFRep(
                sample_rate= 22050,
                f_min=0,
                f_max= int(22050 / 2),
                n_fft = 1024,
                win_length = 1024,
                hop_length = int(0.01 * 22050),
                n_mels = 128
            )
    
    model = ResNet(
        audio_representation = audio_rep
    )
    model.eval()
    with torch.no_grad():
        pred_probs = model(audio_tensor)
    print(pred_probs.shape)
    pred_probs = pred_probs.mean(0,False)
    print(pred_probs.shape)
    pred_labels_and_probs = {LABELS[i]: float(pred_probs[i]) for i in range(len(LABELS))}
    pred_time = round(timer() - start_time, 5)
    return pred_labels_and_probs, pred_time

demo = gr.Interface(fn=predict,
                    inputs=gr.Audio(type="filepath"),
                    outputs=[gr.Label(num_top_classes=23, label="Predictions"), 
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    cache_examples=False
                    )

demo.launch()
