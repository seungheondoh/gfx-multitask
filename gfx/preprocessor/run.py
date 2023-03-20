import os
import ast
import pandas as pd
import numpy as np
import multiprocessing
from gfx.preprocessor.constants import MUSIC_SAMPLE_RATE
from gfx.preprocessor.audio_utils import load_audio, STR_CH_FIRST
from gfx.preprocessor.io_utils import _json_dump

def audio_resampler(_id, audio_path, save_path):
    src, _ = load_audio(
        path= os.path.join(audio_path, _id),
        ch_format= STR_CH_FIRST,
        sample_rate= MUSIC_SAMPLE_RATE,
        downmix_to_mono= True)
    save_name = os.path.join(save_path, _id.replace(".mp3", ".npy"))
    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    np.save(save_name, src.astype(np.float32))
    
def main():
    audio_path= "../../dataset/august/mp3"
    save_path= "../../dataset/august/npy"
    total_track = os.listdir(audio_path)
    print(len(total_track))
    # audio resampling
    pool = multiprocessing.Pool(20)
    pool.starmap(audio_resampler, zip(total_track, [audio_path] * len(total_track), [save_path] * len(total_track)))
    
if __name__ == '__main__':
    main()