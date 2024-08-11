import sys
from time import monotonic
from tqdm import tqdm

import torch

from models.nn_utils import concat_loss_infos
from nn_config import LOG_INTERVAL, CHECKPOINT_STEPS, MODEL_DTYPE


def train_model(train_loader, val_loader, model, optimizer, loss_fn, logger, mixed_precision=False, num_epochs=100, device='cpu'):
    total_steps = 0
    intra_loss_infos = []
    for epoch in range(num_epochs):
        start_time = monotonic()

        train_loss_infos = []
        model.train()
        print("Training...")
        for batch in tqdm(train_loader):
            with torch.autocast('cuda', MODEL_DTYPE):
                frames, actions, states = batch

                frames = frames.to(device)
                actions = actions.to(device)
                states = states.to(device)

                scalar_inputs = torch.concat((actions, states), dim=2)

                # encode and project frames
                enc_frames = model.encode(frames, scalar_inputs)
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
            intra_loss_infos.append(loss_info)

            # if loss_info.total_loss > 0.93 and total_steps > 2000:
            #     import IPython; IPython.embed(); exit(0)

            total_steps += 1

            if len(intra_loss_infos) == LOG_INTERVAL:
                intra_loss_infos = concat_loss_infos(intra_loss_infos)
                logger.log_intra(intra_loss_infos, total_steps)
                intra_loss_infos = []
            if total_steps % CHECKPOINT_STEPS == 0:
                logger.save_model(model, optimizer, 'model', total_steps)

        end_time = monotonic()

        train_loss_infos = concat_loss_infos(train_loss_infos)

        val_loss_infos = []
        model.eval()
        print("Validating...")
        with torch.no_grad():
            for batch in tqdm(val_loader):
                with torch.autocast(device, MODEL_DTYPE):
                    frames, actions, states = batch

                    frames = frames.to(device)
                    actions = actions.to(device)
                    states = states.to(device)
                    
                    if mixed_precision:
                        frames = frames.half()
                        actions = actions.half()
                        states = states.half()

                    scalar_inputs = torch.concat((actions, states), dim=2)

                    # encode and project frames
                    enc_frames = model.encode(frames, scalar_inputs)
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

        val_loss_infos = concat_loss_infos(val_loss_infos)

        logger.log(train_loss_infos, val_loss_infos, epoch)
        logger.log_scalar('utils/elapsed_time', end_time - start_time, epoch)
        logger.log_scalar('utils/sec_per_sample', (end_time - start_time) / len(train_loader.dataset), epoch)