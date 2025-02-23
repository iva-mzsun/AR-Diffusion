import os
import json
import torch
import random
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

from decord import VideoReader
from accelerate.logging import get_logger
logger = get_logger(__name__)

class LatentDataset(Dataset):
    def __init__(self, root, img_mode=False, shuffle=True, max_data_num=None,
                 class_id_file='datasets/ucf101/annotations/classInd.txt'):
        self.img_mode = img_mode
        self.root = root
        # obtain video class ids
        self.class_id = {}
        with open(class_id_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                cid, cname = line.split(' ')
                self.class_id[cname.lower()] = int(cid) - 1

        self.items = [os.path.join(root, item) 
                      for item in os.listdir(root) 
                      if item.endswith('.npy')]
        if shuffle:
            # Note: shuffle latent items before obtain their class labels
            random.shuffle(self.items)
        
        if max_data_num is not None:
            self.items = self.items[:max_data_num]
        
        logger.info(f"The number of total video items: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __skip_sample__(self, idx):
        if idx == len(self.items) - 1:
            return self.__getitem__(0)
        else:
            return self.__getitem__(idx+1)

    def __random_sample__(self):
        idx = np.random.randint(0, len(self.items))
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        curv = self.items[idx]
        # obtain video caption
        videoname = os.path.basename(curv).split('.')[0]
        videoclass = os.path.basename(curv).split('_')[3]
        videoclassid = self.class_id[videoclass.lower()]
        
        # load video latent
        try:
            latent = np.load(curv)
            # assert latent.shape == (16, 12, 1, 32)
        except Exception as e:
            print(f"Failed with Exception: {e}")
            # raise RuntimeError
            return self.__skip_sample__(idx)

        if self.img_mode:
            idx = np.random.randint(0, 16)
            latent = latent[idx:idx+1]

        return dict({
            'class': videoclass, # str
            'class_id': videoclassid, # int
            'video_name': videoname,
            'frames': latent
        })
