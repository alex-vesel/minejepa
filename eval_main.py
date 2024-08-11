from pathlib import Path
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, ConcatDataset

from utils.path_utils import recurse_dir_for_clips
from utils.custom_transforms import *
from utils.datafile import aggregate_data, split_data, idx_to_file_frame
from utils.train import train_model
from utils.logger import Logger
from models import resnet18, resnet10, resnet34, ActionJEPA, VICRegLoss

from nn_config import *
from path_config import *

np.random.seed(0)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else device)

device = 'cpu'

def eval():
    data_path = Path(VPT_DATA_DIR)
    clip_paths = recurse_dir_for_clips(data_path, DOWNSAMPLE_FACTOR)

    train_clips, val_clips, test_clips = split_data(clip_paths)

    # train_clips = ['C:\\Users\\Alex Vesel\\Documents\\vpt\\vpt_data\\10xx\lovely-persimmon-angora-f153ac423f61-20220414-225415']

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
    test_datafiles = aggregate_data(
        clip_paths=train_clips,
        stride=TRAIN_STRIDE,
        num_guard_frames=NUM_GUARD_FRAMES,
        downsample_factor=DOWNSAMPLE_FACTOR,
        cache_dir=VPT_CACHE_DIR,
        transform=composed_transform,
        state_transform=composed_state_transform,
        force_reload=False,
        num_workers=1,
    )

    ## Create dataloaders
    test_loader = DataLoader(
        ConcatDataset(test_datafiles),
        batch_size=128,
        shuffle=True,
        # num_workers=8,
        num_workers=0,
        pin_memory=True,
        # prefetch_factor=2,
        # persistent_workers=True,
    )

    # print(idx_to_file_frame(ConcatDataset(test_datafiles), test_datafiles, 18695))
    # exit()

    ## Define model
    model = ActionJEPA(
        backbone_func=resnet34,
        num_actions=26,
        avg_pool_shape=(3, 3),
        repr_dim=1024,
        projector_output_dim=2048,
    )
    model = model.to(device)

    # load model
    model.load_state_dict(torch.load(MODEL_LOAD_PATH)['model_state_dict'])

    vicreg_loss = VICRegLoss()

    model.eval()
    os.makedirs("high_loss_frames", exist_ok=True)
    for i, batch in enumerate(tqdm(test_loader)):
        frames, actions, states = batch

        frames = frames.float().to(device)
        actions = actions.float().to(device)
        states = states.float().to(device)
        
        scalar_inputs = torch.concat((actions, states), dim=2)

        # encode and project frames
        enc_frames = model.encode(frames, scalar_inputs)
        # proj = model.project(enc_frames)

        # split into current and target frames
        curr_frames = enc_frames[:, 0]
        target = enc_frames[:, 1]

        curr_actions = actions[:, 1]

        # predict next representation
        pred = model.predict(curr_frames, curr_actions)

        # calculate loss
        # loss_info = vicreg_loss(pred, target, proj)

        recon_loss = (pred - target).pow(2).mean(dim=1)
        high_loss_idx = torch.argwhere(recon_loss > 2e-3).squeeze().tolist()

        if type(high_loss_idx) == int:
            high_loss_idx = [high_loss_idx]

        import IPython; IPython.embed(); exit(0)

        # if recon_loss > 0.003:
        #     import IPython; IPython.embed(); exit(0)

        # print high loss indicies
        try:
            if len(high_loss_idx) > 0:
                for idx in high_loss_idx:
                    idx = int(idx)
                    adj_idx = int(idx) + i * 64
                    print(f"{adj_idx} - {float(recon_loss[int(idx)])}")
                    # save those frames
                    plt.imshow(frames[idx, 0, 0].cpu().numpy())
                    plt.savefig(f"high_loss_frames/high_loss_{adj_idx}_0.png")
                    plt.imshow(frames[idx, 1, 0].cpu().numpy())
                    plt.savefig(f"high_loss_frames/high_loss_{adj_idx}_1.png")
                    plt.imshow(frames[idx, 0, 3].cpu().numpy())
                    plt.savefig(f"high_loss_frames/high_loss_{adj_idx}_2.png")
        except:
            import IPython; IPython.embed(); exit(0)




if __name__ == '__main__':
    eval()