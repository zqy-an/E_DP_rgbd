from typing import Dict, Tuple, Union, Callable ,Optional
import copy
from einops import rearrange
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import numpy as np

from cleandiffuser.utils.crop_randomizer import CropRandomizer
from cleandiffuser.nn_condition import BaseNNCondition


def get_norm(channel: int, use_group_norm: bool = True, group_channels: int = 16):
    if use_group_norm:
        return nn.GroupNorm(channel // group_channels, channel, affine=True)
    else:
        return nn.BatchNorm2d(channel, affine=True)

def get_image_coordinates(h, w, normalise):
    x_range = torch.arange(w, dtype=torch.float32)
    y_range = torch.arange(h, dtype=torch.float32)
    if normalise:
        x_range = (x_range / (w - 1)) * 2 - 1
        y_range = (y_range / (h - 1)) * 2 - 1
    image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
    image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
    return image_x, image_y


class ResidualBlock(nn.Module):
    def __init__(
            self, in_channel: int, out_channel: int, downsample: bool = False,
            use_group_norm: bool = True, group_channels: int = 16,
            activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        stride = 2 if downsample else 1

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            get_norm(out_channel, use_group_norm, group_channels), activation,
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            get_norm(out_channel, use_group_norm, group_channels))

        self.skip = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
            get_norm(out_channel, use_group_norm, group_channels)) \
            if downsample else nn.Identity()

    def forward(self, x):
        return self.cnn(x) + self.skip(x)


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax layer as described in [1].
    [1]: "Deep Spatial Autoencoders for Visuomotor Learning"
    Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel

    Adapted from https://github.com/gorosgobe/dsae-torch/blob/master/dsae.py
    """

    def __init__(self, temperature=None, normalise=True):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)) if temperature is None else torch.tensor([temperature])
        self.normalise = normalise

    def forward(self, x):
        n, c, h, w = x.size()
        spatial_softmax_per_map = nn.functional.softmax(x.view(n * c, h * w) / self.temperature, dim=1)
        spatial_softmax = spatial_softmax_per_map.view(n, c, h, w)

        # calculate image coordinate maps
        image_x, image_y = get_image_coordinates(h, w, normalise=self.normalise)
        # size (H, W, 2)
        image_coordinates = torch.cat((image_x.unsqueeze(-1), image_y.unsqueeze(-1)), dim=-1)
        # send to device
        image_coordinates = image_coordinates.to(device=x.device)

        # multiply coordinates by the softmax and sum over height and width, like in [2]
        expanded_spatial_softmax = spatial_softmax.unsqueeze(-1)
        image_coordinates = image_coordinates.unsqueeze(0)
        out = torch.sum(expanded_spatial_softmax * image_coordinates, dim=[2, 3])
        # (N, C, 2)
        return out


class ResNet18(nn.Module):
    def __init__(
            self, image_sz: int, in_channel: int, emb_dim: int, act_fn=lambda: nn.ReLU(),
            use_group_norm: bool = True, group_channels: int = 16,
            use_spatial_softmax: bool = True):
        super().__init__()

        self.image_sz, self.in_channel = image_sz, in_channel

        self.cnn = nn.Sequential(

            # Initial convolution
            nn.Conv2d(in_channel, 64, 7, 2, 3, bias=False),
            get_norm(64, use_group_norm, group_channels),
            act_fn(), nn.MaxPool2d(3, 2, 1),

            # Residual blocks
            ResidualBlock(64, 64, False,
                          use_group_norm, group_channels, act_fn()),
            ResidualBlock(64, 64, False,
                          use_group_norm, group_channels, act_fn()),

            ResidualBlock(64, 128, True,
                          use_group_norm, group_channels, act_fn()),
            ResidualBlock(128, 128, False,
                          use_group_norm, group_channels, act_fn()),

            ResidualBlock(128, 256, True,
                          use_group_norm, group_channels, act_fn()),
            ResidualBlock(256, 256, False,
                          use_group_norm, group_channels, act_fn()),

            ResidualBlock(256, 512, True,
                          use_group_norm, group_channels, act_fn()),
            ResidualBlock(512, 512, False,
                          use_group_norm, group_channels, act_fn()),

            # Final pooling
            nn.AvgPool2d(7, 1, 0) if not use_spatial_softmax else
            SpatialSoftmax(None, True)
        )

        self.mlp = nn.Sequential(
            nn.Linear(np.prod(self.cnn_output_shape), emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim))

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    @property
    def cnn_output_shape(self):
        example = torch.zeros((1, self.in_channel, self.image_sz, self.image_sz),
                              device=self.device, dtype=self.dtype)
        return self.cnn(example).shape

    def forward(self, x):
        feat = self.cnn(x)
        return self.mlp(torch.flatten(feat, 1))


class ResnetObsCondition(BaseNNCondition):
    """
    处理状态、RGB和深度数据的扩散策略条件模块。
    - RGB输入使用TheiaModelDINO处理。
    - 状态和深度输入通过线性层处理。

    输入:
        obs_batch_dict: 包含以下键的观测批次字典:
            - "state": 形状为 (batch_size, seq_len, 25) 的张量
            - "rgb": 形状为 (batch_size, seq_len, 6, 128, 128) 的张量
            - "depth": 形状为 (batch_size, seq_len, 2, 128, 128) 的张量

    输出:
        condition: 形状为 (batch_size, emb_dim) 的张量
    """

    def __init__(
        self,
        state_dim: int = 25,
        cam_num: int = 2,
        use_depth: bool = True,
        use_dino: bool = False,
        emb_dim: int = 256,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.use_depth = use_depth
        self.cam_num = cam_num
        self.state_dim = state_dim
        self.use_dino = use_dino

        # 定义处理状态输入的线性层
        self.state_fc = nn.Sequential(
            nn.Linear(self.state_dim, 64)
        )

        total_feature_dim = emb_dim + 64

        # 最终线性层将连接的特征嵌入到 emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_feature_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # visual_model 配置
        channel = 3 + (1 if self.use_depth else 0) + (3 if self.use_dino else 0)
        in_c = self.cam_num * channel
        self.visual_encoder = ResNet18(
            image_sz = 128, in_channel = in_c, emb_dim = self.emb_dim
        )

    def forward(self, obs_batch_dict: dict, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传递计算条件嵌入。

        参数:
            obs_batch_dict (dict): 包含 "state", "rgb", 和 "depth" 键的字典。
            mask (torch.Tensor, optional): 在本实现中未使用。

        返回:
            torch.Tensor: 形状为 (batch_size, emb_dim) 的条件嵌入。
        """
        # 将所有特征收集到一个列表中
        img_features = []
        # 提取输入
        state = obs_batch_dict["state"]    # 形状: (B, T, 25)
        rgb = obs_batch_dict["rgb"]        # 形状: (B, T, 6, 128, 128)
        # 输入形状
        bs, seq, cn, h, w = rgb.shape
        # 重排并展平前面三个维度
        rgb = rgb.flatten(end_dim=1)    # (B*T, CN, H, W)
        img_features.append(rgb)
        state = state.flatten(end_dim=1) # (B*T, S)

        if self.use_depth:
            depth = obs_batch_dict["depth"]    # 形状: (B, T, N, H, W)
            depth = depth.flatten(end_dim=1)  # (B*T, N, H, W)
            img_features.append(depth)
        if self.use_dino:
            rgb_dino = obs_batch_dict["rgb_dino"] # (B*T, CN, H, W)
            rgb_dino = rgb_dino.flatten(end_dim=1) # (B*T, S)
            img_features.append(rgb_dino)

        # 图像类feature拼接
        img_features = torch.cat(img_features, dim=1)

        # 通过 CNN 处理深度图像
        img_features = self.visual_encoder(img_features)  # (B*S, depth_feature_dim)
        state_features = self.state_fc(state)    # (B*S, state_feature_dim)
        concatenated_features = torch.cat([img_features, state_features], dim=1)  # (B*S, total_feature_dim)

        # 嵌入连接的特征
        embedded = self.mlp(concatenated_features)  # (B*S, emb_dim)
        embedded = embedded.view(bs, -1)

        return embedded

    @torch.no_grad()
    def output_shape(self) -> int:
        """
        计算输出特征的维度。

        返回:
            int: 输出嵌入的维度。
        """
        return self.emb_dim

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
