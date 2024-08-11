import os
import re

def extract_clip_name(x):
    # clip names of style
    # cheeky-cornflower-setter-02e496ce4abb-20220421-093149
    clip_name = re.search(r'([a-z-]+)-([a-z-]+)-([a-z0-9]+)-(\d{8}-\d{6})', x)
    
    if clip_name is None:
        raise ValueError(f"Invalid clip name: {x}")
    
    return clip_name.group(0)


def recurse_dir_for_clips(path, downsample_factor=1):
    # get path to last folder containing mp4 files
    clip_paths = []
    if downsample_factor == 1:
        endswith_tag = '.mp4'
    else:
        endswith_tag = f'_downsampled_{str(downsample_factor).replace(".", "-")}.mp4'
    for root, dirs, files in os.walk(path):
        for file in files:
            # check that file is at least 1MB
            if file.endswith(endswith_tag) and os.stat(os.path.join(root, file)).st_size > 1000000:
                clip_paths.append(root)
                break
    
    return clip_paths