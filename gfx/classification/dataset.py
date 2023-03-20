import os
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Callable, List, Dict, Any
from torch.utils.data import Dataset

class AUGUST_Dataset(Dataset):
    def __init__(self, data_path, split, sr, duration, num_chunks):
        self.data_path = data_path
        self.split = split
        self.sr = sr
        self.input_length = int(sr * duration)
        self.num_chunks = num_chunks
        self.get_split()
        self.get_file_list()
    
    def get_split(self):
        track_split = json.load(open(os.path.join(self.data_path, "august", "track_split.json"), "r"))
        self.train_track = track_split['train_track']
        self.valid_track = track_split['valid_track']
        self.test_track = track_split['test_track']
    
    def get_file_list(self):
        binary = pd.read_csv(os.path.join(self.data_path, "august", "binary.csv"), index_col=0)
        if self.split == "TRAIN":
            self.fl = binary.loc[self.train_track]
        elif self.split == "VALID":
            self.fl = binary.loc[self.valid_track]
        elif self.split == "TEST":
            self.fl = binary.loc[self.test_track]
        else:
            raise ValueError(f"Unexpected split name: {self.split}")
    
    def audio_load(self, _id):
        audio_path = os.path.join(self.data_path, "august", "npy", _id + ".npy")
        audio = np.load(audio_path, mmap_mode='r')
        random_idx = random.randint(0, audio.shape[-1]-self.input_length)
        audio = torch.from_numpy(np.array(audio[random_idx:random_idx+self.input_length]))
        return audio

    def get_train_item(self, index):
        item = self.fl.iloc[index]
        binary = np.array(item.values).astype('float32')
        audio_tensor = self.audio_load(item.name)
        return audio_tensor, binary

    def get_eval_item(self, index):
        item = self.fl.iloc[index]
        binary = np.array(item.values).astype('float32')
        audio_path = os.path.join(self.data_path, "august", "npy", item.name + ".npy")
        audio = np.load(audio_path, mmap_mode='r')
        hop = (len(audio) - self.input_length) // self.num_chunks
        audio = np.stack([np.array(audio[i * hop : i * hop + self.input_length]) for i in range(self.num_chunks)]).astype('float32')
        return {
            "audio":audio, 
            "track_id":item.name, 
            "binary":binary, 
            }

    def __getitem__(self, index):
        if (self.split=='TRAIN') or (self.split=='VALID'):
            return self.get_train_item(index)
        else:
            return self.get_eval_item(index)
            
    def __len__(self):
        return len(self.fl)
