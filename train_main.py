from pathlib import Path

import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, ConcatDataset

from utils.path_utils import recurse_dir_for_clips
from utils.custom_transforms import *
from utils.datafile import aggregate_data, split_data
from utils.train import train_model
from utils.logger import Logger
from models import resnet18, resnet10, resnet34, resnet34x2, resnet50, ActionJEPA, VICRegLoss

from nn_config import *
from path_config import *

np.random.seed(0)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else device)

# device = 'cpu'

## Get clip list
def train():
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
    train_datafiles = aggregate_data(
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

    val_datafiles = aggregate_data(
        clip_paths=val_clips,
        stride=VAL_STRIDE,
        num_guard_frames=NUM_GUARD_FRAMES,
        downsample_factor=DOWNSAMPLE_FACTOR,
        cache_dir=VPT_CACHE_DIR,
        transform=composed_transform,
        state_transform=composed_state_transform,
        force_reload=False,
        num_workers=1,
    )
    
    # val_datafiles = val_datafiles[:3]
    
    train_dataset = ConcatDataset(train_datafiles)
    val_dataset = ConcatDataset(val_datafiles)
    
    # indices = list(range(len(train_dataset)))
    # np.random.shuffle(indices)
    
    # train_dataset = torch.utils.data.Subset(train_dataset, indices[:131072])
    # train_dataset = torch.utils.data.Subset(train_dataset, indices[:128])

    ## Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    print(f"Total train clips: {len(train_clips)}, total train frames: {len(train_loader.dataset)}")
    print(f"Total val clips: {len(val_clips)}, total val frames: {len(val_loader.dataset)}")

    ## Define model
    model = ActionJEPA(
        backbone_func=resnet34,
        num_actions=26,
        avg_pool_shape=(3, 3),
        # avg_pool_shape=(1, 1),
        repr_dim=1024,
        projector_output_dim=2048,
    )
    model = model.to(device)

    # print number of trainable parameters
    print(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    vicreg_loss = VICRegLoss()

    ## Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if RELOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH)['model_state_dict'])
        optimizer.load_state_dict(torch.load(MODEL_LOAD_PATH)['optimizer_state_dict'])
        for g in optimizer.param_groups:
            g['lr'] = LEARNING_RATE
            g['weight_decay'] = WEIGHT_DECAY
        print(f"Model loaded from {MODEL_LOAD_PATH}")
        
    else:
        print("Training from scratch...")

    ## Train model
    logger = Logger(logdir=EXPERIMENT_RESULTS_DIR)

    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=vicreg_loss,
        logger=logger,
        mixed_precision=MIXED_PRECISION,
        num_epochs=TRAIN_EPOCHS,
        device=device,
    )


if __name__ == '__main__':
    train()