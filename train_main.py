from pathlib import Path

import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, ConcatDataset

from utils.path_utils import recurse_dir_for_clips
from utils.custom_transforms import *
from utils.datafile import aggregate_data, split_data
from utils.train import train_model
from utils.logger import Logger
from models import resnet18, resnet10, ActionJEPA, VICRegLoss

from nn_config import *
from path_config import *

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else device)

## Get clip list
data_path = Path(VPT_DATA_DIR)
clip_paths = recurse_dir_for_clips(data_path)

train_clips, val_clips, test_clips = split_data(clip_paths)

## Define transforms
transform_list = [              # (3, 360, 640, 3)
    DifferenceOpticalFlow(),    # (2, 360, 640, 5)
    MoveAxis(3, 1),             # (2, 5, 640, 360)
    DivideByScalar(255, axis=1, channels=[0, 1, 2]),
    DivideByScalar(20, axis=1, channels=[3, 4]),
]

composed_transform = Compose(transform_list)

## Aggregate data
train_datafiles = aggregate_data(
    clip_paths=train_clips,
    stride=1,
    cache_dir=VPT_CACHE_DIR,
    transform=composed_transform,
    force_reload=True,
    num_workers=1,
)

val_datafiles = aggregate_data(
    clip_paths=val_clips,
    stride=1,
    cache_dir=VPT_CACHE_DIR,
    transform=composed_transform,
    force_reload=True,
    num_workers=1,
)

## Create dataloaders
train_loader = DataLoader(
    ConcatDataset(train_datafiles),
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_loader = DataLoader(
    ConcatDataset(val_datafiles),
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

## Define model
model = ActionJEPA(
    backbone_func=resnet10,
    num_actions=26,
    avg_pool_shape=(1, 1),
)
model = model.to(device)

vicreg_loss = VICRegLoss()

## Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# print number of trainable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_params}')

## Train model
logger = Logger(logdir=EXPERIMENT_RESULTS_DIR)

train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    optimizer=optimizer,
    loss_fn=vicreg_loss,
    logger=logger,
    num_epochs=TRAIN_EPOCHS,
    device=device,
)