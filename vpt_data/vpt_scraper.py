import requests
import multiprocessing
import json
import os

OUT_DIR = 'vpt_data/10xx/'

os.makedirs(OUT_DIR, exist_ok=True)

with open('vpt_data/all_10xx_Jun_29.json', 'r') as f:
    links = json.load(f)

basedir = links['basedir']

def download_clip(link):
    clip_id = link.split('/')[-1].split('.')[0]
    video_url = basedir + link
    action_url = basedir + link.replace('mp4', 'jsonl')

    clip_path = os.path.join(OUT_DIR, clip_id)
    os.makedirs(clip_path, exist_ok=True)

    video_path = os.path.join(clip_path, 'video.mp4')
    action_path = os.path.join(clip_path, 'action.jsonl')

    # check if video path is of meaningful size (>1MB)
    if os.path.exists(video_path) and os.stat(video_path).st_size > 1000000:
        print(f"Skipping {clip_id}")
    else:
        with open(action_path, 'wb') as f:
            f.write(requests.get(action_url).content)

        with open(video_path, 'wb') as f:
            r = requests.get(video_url)
            print(r.status_code)
            f.write(r.content)
        print(f"Downloaded {clip_id}")



if __name__ == '__main__':
    pool = multiprocessing.Pool(1)
    pool.map(download_clip, links['relpaths'])
    pool.close()


