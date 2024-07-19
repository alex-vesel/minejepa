import os
import cv2
import torch
import json
import multiprocessing
import numpy as np
from pathlib import Path
from collections import OrderedDict

from torch.utils.data import Dataset

from .path_utils import extract_clip_name, recurse_dir_for_clips
from .vpt_utils import ACTION_MAP, vpt_dpixels_to_degrees
from .custom_transforms import *


class DataFile(Dataset):
    def __init__(self,
                 clip_path: str,
                 stride: int = 1,
                 transform=None,
                 cache_dir: str = None,
                 force_reload: bool = False
                ):

        self.clip_path = clip_path
        self.clip_name = extract_clip_name(self.clip_path)
        self.stride = stride
        self.transform = transform
        self.cache_dir = cache_dir
        self.force_reload = force_reload

        self.cap = cv2.VideoCapture(os.path.join(self.clip_path, 'video.mp4'))
        self.cache_path_action = Path(os.path.join(self.cache_dir, f'{self.clip_name}_action.bin'))

        self._load_data()


    def _load_data(self):
        # # check if memmaps made for this clip
        if not self.cache_path_action.exists() or self.force_reload:
            with open(os.path.join(self.clip_path, 'action.jsonl'), 'r') as f:
                actions = f.readlines()

            # dy_actions = []
            # d_degrees = []

            # for action in actions:
            #     if "key.keyboard.mouse.left" in action:
            #         import IPython; IPython.embed(); exit(0)
            #     action = json.loads(action)
            #     print(action['mouse']['buttons'])

            processed_actions = []
            # prev_action = None
            for action in actions:
                action = json.loads(action)
                processed_action = OrderedDict.fromkeys(ACTION_MAP.keys(), 0)
                # if prev_action is not None:
                #     print(prev_action['mouse']['dy'], action['pitch'] - prev_action['pitch'])
                #     dy_actions.append(prev_action['mouse']['dx'])
                #     d_degrees.append(action['yaw'] - prev_action['yaw'])
                # prev_action = action
                # fill in action from keypresses
                processed_action['camera_pitch'] = vpt_dpixels_to_degrees(action['mouse']['dy']) / 180
                processed_action['camera_yaw'] = vpt_dpixels_to_degrees(action['mouse']['dx']) / 180
                for key, value in ACTION_MAP.items():
                    if value in action['keyboard']['keys']:
                        processed_action[key] = 1

                if 0 in action['mouse']['buttons']:
                    processed_action['attack'] = 1
                if 1 in action['mouse']['buttons']:
                    processed_action['use'] = 1

                if 'key.keyboard.f3' in action['keyboard']['newKeys']:
                    processed_action['f3'] = 1

                processed_actions.append(list(processed_action.values()))

            processed_actions = np.array(processed_actions)
            self.action_memmap = np.memmap(self.cache_path_action, dtype='float32', mode='w+', shape=processed_actions.shape)
            self.action_memmap = processed_actions
        else:
            self.action_memmap = np.memmap(self.cache_path_action, dtype='float32', mode='r')

        # dy_actions = np.array(dy_actions)
        # d_degrees = np.array(d_degrees)

        # dy_actions = dy_actions[d_degrees != 0]
        # d_degrees = d_degrees[d_degrees != 0]

        # # fit line
        # m, b = np.polyfit(dy_actions, d_degrees, 1)


    def __len__(self):
        return self.action_memmap.shape[0] - 2


    def __getitem__(self, idx):
        frames = np.array(self._get_frames(range(idx, idx + 3)))
        actions = self.action_memmap[idx:idx+2]

        if self.transform:
            frames = self.transform(frames)

        return frames, actions


    def _get_frames(self, range):
        frames = []
        for i in range:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)

        return frames


def aggregate_data(clip_paths, stride=1, cache_dir=None, transform=None,
                   force_reload=False, num_workers=1):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    data = []
    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            pool.starmap(DataFile, [(clip_path, stride, transform, cache_dir, force_reload) for clip_path in clip_paths])
    else:
        for clip_path in clip_paths:
            datafile = DataFile(clip_path, stride, transform, cache_dir, force_reload)
            data.append(datafile)

    return data


def split_data(datafiles, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    np.random.shuffle(datafiles)
    n = len(datafiles)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = datafiles[:train_end]
    val_data = datafiles[train_end:val_end]
    test_data = datafiles[val_end:]

    return train_data, val_data, test_data


if __name__ == '__main__':
    from torchvision.transforms import Compose

    data_path = Path('./vpt_data')
    clip_paths = recurse_dir_for_clips(data_path)

    transform_list = [
        Reshape((2, 3, 360, 640)),
        DivideByScalar(255),
    ]

    composed_transform = Compose(transform_list)

    datafiles = aggregate_data(
        clip_paths=clip_paths,
        stride=1,
        cache_dir='./vpt_cache',
        transform=composed_transform,
        force_reload=True,
        num_workers=1,
    )
