import os
import cv2
import torch
import json
import ffmpeg
import sys
import multiprocessing
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from .path_utils import extract_clip_name, recurse_dir_for_clips
from .vpt_utils import ACTION_MAP, vpt_dpixels_to_degrees
from .profile_utils import profile
from .custom_transforms import *


class DataFile(Dataset):
    def __init__(self,
                 clip_path: str,
                 stride: int = 1,
                 num_guard_frames: int = 0,
                 transform=None,
                 state_transform=None,
                 downsample_factor: float = 1,
                 cache_dir: str = None,
                 force_reload: bool = False,
                 cache_method: str = "vid",  # options are npz, png, or vid
                ):

        self.clip_path = clip_path
        self.clip_name = extract_clip_name(self.clip_path)
        self.stride = stride
        self.num_guard_frames = num_guard_frames
        self.transform = transform
        self.state_transform = state_transform
        self.downsample_factor = downsample_factor
        self.cache_dir = cache_dir
        self.force_reload = force_reload
        self.cache_method = cache_method

        self.cache_path_action = Path(os.path.join(self.cache_dir, f'{self.clip_name}_action.bin'))
        self.cache_path_state = Path(os.path.join(self.cache_dir, f'{self.clip_name}_state.bin'))

        self._load_data()


    def _load_data(self):
        # # check if memmaps made for this clip
        if not self.cache_path_action.exists() or not self.cache_path_state.exists() or self.force_reload:
            with open(os.path.join(self.clip_path, 'action.jsonl'), 'r') as f:
                actions = f.readlines()

            cap = cv2.VideoCapture(os.path.join(self.clip_path, 'video.mp4'))

            processed_actions = []
            processed_states = []
            # prev_action = None
            for idx, action in enumerate(actions):
                try:
                    action = json.loads(action)
                except:
                    # remove all mp4s from clip_path
                    os.system(f'del {self.clip_path}\\video.mp4')
                    os.system(f'del {self.clip_path}\\video_downsampled_2.mp4')
                    os.system(f'del {self.clip_path}\\video_downsampled_2-8125.mp4')
                    print('removed')
                    return
                processed_action = OrderedDict.fromkeys(ACTION_MAP.keys(), 0)

                # fill in action from keypresses
                processed_action['camera_pitch'] = vpt_dpixels_to_degrees(action['mouse']['dy']) / 180
                processed_action['camera_yaw'] = vpt_dpixels_to_degrees(action['mouse']['dx']) / 180
                for key, value in ACTION_MAP.items():
                    if value != "key.keyboard.e" and value in action['keyboard']['keys']:
                        processed_action[key] = 1
                    elif value == "key.keyboard.e" and value in action['keyboard']['newKeys']:
                        processed_action[key] = 1

                if 0 in action['mouse']['buttons']:
                    processed_action['attack'] = 1
                if 1 in action['mouse']['buttons']:
                    processed_action['use'] = 1
                if 2 in action['mouse']['buttons']:
                    processed_action['pickitem'] = 1

                if 'key.keyboard.f3' in action['keyboard']['newKeys']:
                    processed_action['f3'] = 1

                processed_actions.append(list(processed_action.values()))
                processed_states.append([action['mouse']['x'], action['mouse']['y']])

            processed_actions = np.array(processed_actions)
            self.action_memmap = np.memmap(self.cache_path_action, dtype='float32', mode='w+', shape=processed_actions.shape)
            self.action_memmap[:] = processed_actions[:]
            self.action_memmap.flush()
            
            processed_states = np.array(processed_states)
            self.state_memmap = np.memmap(self.cache_path_state, dtype='float32', mode='w+', shape=processed_states.shape)
            self.state_memmap[:] = processed_states[:]
            self.state_memmap.flush()
        else:
            self.action_memmap = np.memmap(self.cache_path_action, dtype='float32', mode='r').reshape(-1, 26)
            self.state_memmap = np.memmap(self.cache_path_state, dtype='float32', mode='r').reshape(-1, 2)

        # create npz cache for each frame if required
        if self.cache_method == "npz" and (self.force_reload or not os.path.exists(os.path.join(self.cache_dir, f'{self.clip_name}_0.npz'))):
            cap = cv2.VideoCapture(os.path.join(self.clip_path, f'video_downsampled_{self.downsample_factor}.mp4'))
            frames = self._get_frames_cap(range(0, len(self.action_memmap)), cap)
            frames = np.array(frames)
            for i, frame in enumerate(frames):
                np.savez_compressed(os.path.join(self.cache_dir, f'{self.clip_name}_{i}.npz'), frame=frame)

        # create png cache for each frame if required
        if self.cache_method == "png" and (self.force_reload or not os.path.exists(os.path.join(self.cache_dir, f'{self.clip_name}_0.png'))):
            cap = cv2.VideoCapture(os.path.join(self.clip_path, f'video_downsampled_{self.downsample_factor}.mp4'))
            frames = self._get_frames_cap(range(0, len(self.action_memmap)), cap)
            for i, frame in enumerate(frames):
                cv2.imwrite(os.path.join(self.cache_dir, f'{self.clip_name}_{i}.png'), frame)


    def __len__(self):
        return (self.action_memmap.shape[0] - 2 - self.num_guard_frames) // self.stride


    def __getitem__(self, idx):
        idx = idx * self.stride + self.num_guard_frames

        if self.cache_method == "npz":
            frames = self._get_frames_npz(range(idx, idx + 3))
        elif self.cache_method == "png":
            frames = self._get_frames_png(range(idx, idx + 3))
        else:
            cap = cv2.VideoCapture(os.path.join(self.clip_path, f'video_downsampled_{self.downsample_factor}.mp4'))
            frames = self._get_frames_cap(range(idx, idx + 3), cap)
        actions = self.action_memmap[idx:idx+2]
        states = self.state_memmap[idx:idx+3]

        if self.transform:
            frames = self.transform(frames)
            
        if self.state_transform:
            states = self.state_transform(states)

        return frames, actions, states#, self.clip_name


    def _get_frames_ffmpeg(self, range):
        frames = []
        for i in range:
            out, _ = (
                ffmpeg
                .input(os.path.join(self.clip_path, f'video_downsampled_{self.downsample_factor}.mp4'), ss=i)
                .filter_('select', 'gte(n,{})'.format(i))
                .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
                .run(capture_stdout=True)
            )
            frame = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
            frames.append(frame)
        frames = np.array(frames)

        return frames

    def _get_frames_cap(self, range, cap):
        frames = []
        for i in range:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        frames = np.array(frames)

        return frames


    def _get_frames_npz(self, range):
        frames = []
        for i in range:
            frame = np.load(os.path.join(self.cache_dir, f'{self.clip_name}_{i}.npz'))['frame']
            frames.append(frame)
        frames = np.array(frames)

        return frames


    def _get_frames_png(self, range):
        frames = []
        for i in range:
            frame = cv2.imread(os.path.join(self.cache_dir, f'{self.clip_name}_{i}.png'))
            frames.append(frame)
        frames = np.array(frames)

        return frames


def aggregate_data(clip_paths, stride=1, num_guard_frames=0, downsample_factor=1, cache_dir=None, transform=None,
                   state_transform=None, force_reload=False, num_workers=1):
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    data = []
    if num_workers > 1:
        # iterate over datafiles in parallel
        with multiprocessing.Pool(num_workers) as pool:
            for datafile in tqdm(pool.starmap(DataFile, [(clip_path, stride, num_guard_frames, transform, state_transform, downsample_factor, cache_dir, force_reload) for clip_path in clip_paths])):
                if len(datafile) > 0:
                    data.append(datafile)
    else:
        for clip_path in tqdm(clip_paths):
            datafile = DataFile(clip_path, stride, num_guard_frames, transform, state_transform, downsample_factor, cache_dir, force_reload)
            if len(datafile) > 0:
                data.append(datafile)

    return data


def split_data(datafiles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    np.random.shuffle(datafiles)
    n = len(datafiles)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = datafiles[:train_end]
    val_data = datafiles[train_end:val_end]
    test_data = datafiles[val_end:]

    return train_data, val_data, test_data


def idx_to_file_frame(dataset, datafiles, idx):
    # takes ordered index and returns the file and frame index
    cum_sizes = dataset.cumulative_sizes
    for i, size in enumerate(cum_sizes):
        if idx < size:
            return datafiles[i].clip_name, idx - cum_sizes[i-1] if i > 0 else idx


if __name__ == '__main__':
    import sys
    from torchvision.transforms import Compose
    sys.path.append('.')
    from path_config import *
    from nn_config import *

    data_path = Path(VPT_DATA_DIR)
    clip_paths = recurse_dir_for_clips(data_path, DOWNSAMPLE_FACTOR)

    train_clips, val_clips, test_clips = split_data(clip_paths)

    # train_clips = train_clips[:2]
    # val_clips = val_clips[10:20]

    ## Define transforms
    transform_list = [              # (3, 180, 320, 3)
        # DifferenceOpticalFlow(),    # (2, 180, 320, 5)
        FloatTransform(),
        ConcatPreviousFrame(),     # (2, 180, 320, 6)
        MoveAxis(3, 1),             # (2, 5, 180, 320)
        DivideByScalar(255.),
    ]

    composed_transform = Compose(transform_list)
    
    state_transform_list = [
        FloatTransform(),
        ConcatPreviousFrame(),
        DivideByScalar(3000.)
    ]
    
    composed_state_transform = Compose(state_transform_list)

    ## Aggregate data
    train_clips = train_clips[:4]
    train_datafiles = aggregate_data(
        clip_paths=train_clips,
        stride=TRAIN_STRIDE,
        num_guard_frames=NUM_GUARD_FRAMES,
        downsample_factor=DOWNSAMPLE_FACTOR,
        cache_dir=VPT_CACHE_DIR,
        transform=composed_transform,
        state_transform=composed_state_transform,
        force_reload=True,
        num_workers=1,
    )

    # iterate over a few datafiles
    from time import monotonic

    start = monotonic()

    for i in range(1):
        datafile = train_datafiles[i]
        for j in tqdm(range(len(datafile))):
            frames, actions, states = datafile[j]

    end = monotonic()
    print(end - start)
