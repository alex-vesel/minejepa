from typing import List

import torch
import torch.nn as nn

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1], bias=False))

    return nn.Sequential(*layers)


class Projector(nn.Module):
    def __init__(self, layers_dims):
        super(Projector, self).__init__()
        self.layers = build_mlp(layers_dims)

    def forward(self, x):
        return self.layers(x)


class ActionJEPA(nn.Module):
    def __init__(self, backbone_func, num_actions, avg_pool_shape=(1, 1), projector_output_dim=8192):
        super(ActionJEPA, self).__init__()
        self.backbone, self.num_features = backbone_func(avg_pool_shape=avg_pool_shape)
        
        f_predict = (
            [self.num_features + num_actions] +
            [2 * self.num_features] +
            [self.num_features]
        )
        self.predict_head = Projector(f_predict)

        f_action_fuse = (
            [self.num_features + num_actions] +
            [2 * self.num_features] +
            [self.num_features]
        )
        self.action_fuse_head = Projector(f_action_fuse)

        f_proj = (
            [self.num_features] +
            [2 * self.num_features] +
            [projector_output_dim]
        )
        self.projector = Projector(f_proj)


    def predict(self, rep, a):
        x = torch.cat([rep, a], dim=1)
        x = self.predict_head(x)

        return x


    def encode(self, x, a):
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        a = a.view(-1, a.shape[-1])

        x = self.backbone(x)
        x = torch.cat([x, a], dim=1)
        x = self.action_fuse_head(x)

        x = x.view(x_shape[0], x_shape[1], -1)
        
        return x
    

    def project(self, rep):
        rep = rep.view(-1, rep.shape[-1])
        return self.projector(rep)