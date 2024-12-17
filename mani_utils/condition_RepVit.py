from typing import Dict, Tuple, Union, Callable ,Optional
import copy
from einops import rearrange
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from cleandiffuser.utils.crop_randomizer import CropRandomizer
from cleandiffuser.nn_condition import BaseNNCondition

from timm.models.layers import SqueezeExcite

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv

class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

from timm.models.vision_transformer import trunc_normal_
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self):
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier


class RepViT(nn.Module):
    def __init__(self, cfgs, image_channel = 16, num_classes=1000, distillation=False):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(image_channel, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                                          Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        self.classifier = Classfier(output_channel, num_classes, distillation)

    def forward(self, x):
        # x = self.features(x)
        for f in self.features:
            x = f(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.classifier(x)
        return x

#
# class TheiaModelDINO(nn.Module):
#     """
#     TheiaModelDINO 类，用于处理 RGB 图像并提取特征。
#
#     输入:
#         images: Tensor, 形状为 (batch_size, seq_len, C, H, W) 或 (batch_size, C, H, W)
#
#     输出:
#         fused_features: Tensor, 提取并融合后的特征
#     """
#
#     def __init__(self, theia_model_name: str, target_model_name: str, device: torch.device):
#         super(TheiaModelDINO, self).__init__()
#         self.device = device
#
#         # 加载预训练的 TheiaModel
#         self.theia_model = AutoModel.from_pretrained(theia_model_name, trust_remote_code=True)
#         self.theia_model.to(self.device)
#         self.theia_model.eval()  # 设置为评估模式
#         self.target_model_name = target_model_name
#
#         # 定义一个简单的下采样层（可选）
#         # self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
#
#     def forward(self, images: torch.Tensor) -> torch.Tensor:
#         with torch.no_grad():
#             # 获取特征输出 dino (bs,1024,16,16)
#             features = self.theia_model(images)[self.target_model_name].detach().cpu()
#             # dino_gt_feature = rearrange(features, "b c h w -> b (h w) c")
#             dino_gt_dec2, pca = decode_dinov2(features, pca=None)
#             # 先将通道维度移到第二维度
#             tensor = dino_gt_dec2.permute(0, 3, 1, 2).to(self.device)
#             # 使用 interpolate 调整大小
#             tensor_resized = F.interpolate(tensor, size=(128, 128), mode='bilinear', align_corners=False)
#
#         # 可选的下采样操作
#         # features = self.downsample(features)
#         return tensor_resized

class RepVitObsCondition(BaseNNCondition):
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

        # 基类初始化后再初始化 TheiaModelDINO
        # self.rgb_model = TheiaModelDINO(
        #     theia_model_name=theia_model_name,
        #     target_model_name=target_model_name,
        #     device=self.device
        # )

        # visual_model 配置
        cfgs = [
            [3, 2, 40, 1, 0, 1],
            [3, 2, 40, 0, 0, 1],
            [3, 2, 80, 0, 0, 2],
            [3, 2, 80, 1, 0, 1],
            [3, 2, 80, 0, 0, 1],
            [3, 2, 160, 0, 1, 2],
            [3, 2, 160, 1, 1, 1],
            [3, 2, 160, 0, 1, 1],
            [3, 2, 160, 1, 1, 1],
            [3, 2, 160, 0, 1, 1],
            [3, 2, 160, 1, 1, 1],
            [3, 2, 160, 0, 1, 1],
            [3, 2, 160, 1, 1, 1],
            [3, 2, 160, 0, 1, 1],
            [3, 2, 160, 0, 1, 1],
            [3, 2, 320, 0, 1, 2],
            [3, 2, 320, 1, 1, 1],
        ]
        channel = 3 + (1 if self.use_depth else 0) + (3 if self.use_dino else 0)
        in_c = self.cam_num * channel
        self.visual_encoder = RepViT(cfgs, image_channel=in_c, num_classes=emb_dim, distillation=False)

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
        # 通过 Vit 处理深度图像
        img_features = self.visual_encoder(img_features)  # 形状: (B*S, depth_feature_dim)

        # 通过线性层处理状态
        state_features = self.state_fc(state)    # 形状: (B*S, state_feature_dim)

        # 连接所有特征
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
