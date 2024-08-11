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
    def __init__(self, backbone_func, num_actions, avg_pool_shape=(1, 1), repr_dim=512, projector_output_dim=1024):
        super(ActionJEPA, self).__init__()
        self.backbone, self.num_features = backbone_func(avg_pool_shape=avg_pool_shape)

        self.repr_dim = repr_dim

        f_predict = (
            [self.repr_dim + num_actions] +
            [2 * self.repr_dim] +
            [4 * self.repr_dim] +
            [self.repr_dim]
        )
        self.predict_head = Projector(f_predict)

        f_feature_fuse = (
            [self.num_features + num_actions + 4] +
            [self.repr_dim] +
            [self.repr_dim] +
            [self.repr_dim]
        )
        self.feature_fuse_head = Projector(f_feature_fuse)

        f_proj = (
            [self.repr_dim] +
            [2 * projector_output_dim] +
            [projector_output_dim]
        )
        self.projector = Projector(f_proj)

        # get num of params in each model
        # start with backbone
        print(f"Backbone params: {sum(p.numel() for p in self.backbone.parameters())}")
        print(f"Predict head params: {sum(p.numel() for p in self.predict_head.parameters())}")
        print(f"Feature fuse head params: {sum(p.numel() for p in self.feature_fuse_head.parameters())}")
        print(f"Projector params: {sum(p.numel() for p in self.projector.parameters())}")



    def predict(self, rep, a):
        x = torch.cat([rep, a], dim=1)
        x = self.predict_head(x)

        return x


    def encode(self, x, a):
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        a = a.view(-1, a.shape[-1])

        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = torch.cat([x, a], dim=1)
        x = self.feature_fuse_head(x)

        x = x.view(x_shape[0], x_shape[1], -1)
        
        return x
    

    def project(self, rep):
        rep = rep.view(-1, rep.shape[-1])
        return self.projector(rep)