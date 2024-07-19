import sys
from time import monotonic

import torch

from models.nn_utils import concat_loss_infos


def train_model(train_loader, val_loader, model, optimizer, loss_fn, logger, num_epochs=100, device='cpu'):
    for epoch in range(num_epochs):
        start_time = monotonic()

        train_loss_infos = []
        model.train()
        for batch in train_loader:
            frames, actions = batch

            frames = frames.float().to(device)
            actions = actions.float().to(device)

            # encode and project frames
            enc_frames = model.encode(frames, actions)
            proj = model.project(enc_frames)
            
            # split into current and target frames
            curr_frames = enc_frames[:, 0]
            target = enc_frames[:, 1]

            curr_actions = actions[:, 1]

            # predict next representation
            pred = model.predict(curr_frames, curr_actions)

            # calculate loss
            loss_info = loss_fn(pred, target, proj)
            loss = loss_info.total_loss

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_infos.append(loss_info)

            if len(train_loss_infos) > 2:
                break

        end_time = monotonic()

        train_loss_infos = concat_loss_infos(train_loss_infos)

        val_loss_infos = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                frames, actions = batch

                frames = frames.float().to(device)
                actions = actions.float().to(device)

                # encode and project frames
                enc_frames = model.encode(frames, actions)
                proj = model.project(enc_frames)

                # split into current and target frames
                curr_frames = enc_frames[:, 0]
                target = enc_frames[:, 1]

                curr_actions = actions[:, 1]

                # predict next representation
                pred = model.predict(curr_frames, curr_actions)

                # calculate loss
                loss_info = loss_fn(pred, target, proj)
                val_loss_infos.append(loss_info)

                if len(val_loss_infos) > 2:
                    break

        val_loss_infos = concat_loss_infos(val_loss_infos)

        logger.log(train_loss_infos, val_loss_infos, epoch)
        logger.log_scalar('utils/elapsed_time', end_time - start_time, epoch)
        logger.log_scalar('utils/sec_per_sample', (end_time - start_time) / len(train_loader.dataset), epoch)