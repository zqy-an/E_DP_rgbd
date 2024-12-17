import torch.nn as nn
from typing import Dict, Tuple, Union, Callable ,Optional
import copy
from einops import rearrange
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from cleandiffuser.utils.crop_randomizer import CropRandomizer
from cleandiffuser.nn_condition import BaseNNCondition


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


class PlainConv(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_dim=256,
        pool_feature_map=False,
        last_act=True,  # True for ConvBody, False for CNN
    ):
        super().__init__()
        # assume input image size is 64x64

        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(128, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        if pool_feature_map:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=last_act)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4 * 4, [out_dim], last_act=last_act)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class CNNObsCondition(BaseNNCondition):
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
        self.visual_encoder = PlainConv(
            in_channels=in_c, out_dim=self.emb_dim, pool_feature_map=True
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
