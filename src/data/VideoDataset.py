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

class VideoDataset(Dataset):
    def __init__(self, root, extract_fps, clip_nframe, 
                 ids_file=None, id2cap_file=None, fix_prompt=None,
                 shuffle=True, size=None, max_data_num=None):
        self.size = size
        self.root = root
        self.fix_prompt = fix_prompt
        self.extract_fps = extract_fps
        self.clip_nframe = clip_nframe
        self.interpolation = Image.BICUBIC

        # load id2cap dict
        if id2cap_file is None:
            self.id2cap = None
        elif id2cap_file.endswith('json'):
            self.id2cap = json.load(open(id2cap_file))
        else:
            raise NotImplementedError
        # load data
        if ids_file is None:
            videos = sorted(glob(os.path.join(root, "**/*.avi"))) \
                + sorted(glob(os.path.join(root, "**/*.mp4"))) \
                + sorted(glob(os.path.join(root, "*.mp4")))
        else:
            videos = sorted(json.load(open(ids_file)))
            videos = [os.path.join(root, item) for item in videos 
                      if os.path.exists(os.path.join(root, item))]
        self.items = videos

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size)
        ])

        if shuffle:
            print("- NOTE: shuffle video items!")
            from random import shuffle
            shuffle(self.items)
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

    def get_frames(self, curv):
        # read video
        # path = os.path.join(self.root, curv)
        path = curv
        container = VideoReader(path)
        fps = container.get_avg_fps()
        nframe = len(container)
        nframe_total = self.clip_nframe
        if nframe < self.clip_nframe:
            raise RuntimeError(f"No enough total frames for {curv}")

        start = random.choice(range(nframe - nframe_total + 1))
        frame_skip = max(1, (nframe_total - 1) // self.clip_nframe)
        clip_indexes = range(0, nframe_total, frame_skip)
        clip_indexes = clip_indexes[:self.clip_nframe]
        sample_idx = [i + start for i in clip_indexes]
        # print(clip_indexes, sample_idx)
        container.skip_frames(1)
        frames = container.get_batch(sample_idx).asnumpy()
        frames = torch.from_numpy(frames.transpose(0, 3, 1, 2))
        frames = frames.to(torch.float32) / 255.0 # NOTE: The image tensors are in [0, 1] for 1d tokenizer
        frames = self.transform(frames)
        return frames

    def __getitem__(self, idx):
        curv = self.items[idx]
        # obtain video caption
        if self.id2cap is None:
            videoname = "vnull"
            prompt = self.fix_prompt
        else:
            videoname = os.path.basename(curv)
            prompt = self.id2cap[curv]

        # load video frames
        try:
            frames = self.get_frames(curv)
            assert frames.shape == (self.clip_nframe, 3, 256, 256), f'Invalid frame shape {frames.shape}'
        except Exception as e:
            print(f"Failed with Exception: {e}")
            # raise RuntimeError
            return self.__skip_sample__(idx)

        return dict({
            'txt': prompt, # str
            'video_name': videoname,
            'frames': frames
        })
