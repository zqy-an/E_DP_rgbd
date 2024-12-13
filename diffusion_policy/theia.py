import os
import torch
from PIL import Image
import numpy as np
from transformers import AutoModel

from theia.decoding import  prepare_depth_decoder, prepare_mask_generator, decode_everything

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

from typing import Dict, Tuple, Union, Callable ,Optional
import copy
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from transformers import AutoModel



def denormalize_feature(
    x: torch.Tensor, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Denormalize the features using mean and std.

    Args:
        x (torch.Tensor): features to be denomalized.
        mean (Optional[torch.Tensor], optional): mean value of the features. Defaults to None
        std (Optional[torch.Tensor], optional): std value of the features. Defaults to None.

    Returns:
        torch.Tensor: denormalized features.
    """
    if mean is None and std is None:
        return x
    elif mean is None and std is not None:
        return x * std
    elif mean is not None and std is None:
        return x + mean
    return x * std + mean


def load_theia_feature_stats(
    feature_model_name: str, stat_file_root: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the statistics (mean and variance) of the features, per model.

    Args:
        feature_models (list[str]): names of the models. Note: there are `/` in the name.
        stat_file_root (str): directory that holds feature stat files.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: means and variance.
    """

    model_name = feature_model_name.replace("/", "_")
    feature_means = torch.from_numpy(
        np.load(os.path.join(stat_file_root, f"imagenet_mean_{model_name}.npy"))
    )
    feature_vars = torch.from_numpy(np.load(os.path.join(stat_file_root, f"imagenet_var_{model_name}.npy")))
    return feature_means, feature_vars

def decode_dinov2(
    features: NDArray, threshold: int | float = -100, interpolation: bool = False, pca: Optional[PCA] = None
) -> tuple[NDArray, PCA]:
    """
    Decode the input `features` in DINOv2 style using PCA.

    Args:
        features (NDArray): features to be decoded, should be in shape [batch_size, num_tokens, latent_dim].
        threshold (int | float): threshold of foreground-background split in PCA visualization.
            Defaults to -100 (all patches are included).
        interpolation (bool): whether interpolate the 16x16 pca map to the original image size.
        pca (Optional[PCA]): if provided, use the provided PCA. This is to keep visualizations stable across samples.

    Returns:
        tuple[NDArray, PCA]: the rendered image of this visualization, in NDArray in size
            [batch_size, height, width, channels] with value ranges [0, 1], and the PCA used in this visualization.
    """
    features = features.numpy()
    batch_size, spatial_size, latent_dim = features.shape
    h = w = int(spatial_size**0.5)

    features = features.reshape(-1, latent_dim)

    if pca is None:
        pca = PCA(n_components=3)
        pca.fit(features)

    pca_features = pca.transform(features)

    # segment using the first component
    bg_mask = pca_features[:, 0] < threshold
    fg_mask = ~bg_mask

    # PCA for only foreground patches
    # pca.fit(features[fg_mask])
    pca_features_fg = pca.transform(features[fg_mask])
    for i in range(3):
        pca_features_fg[:, i] = minmax_scale(pca_features_fg[:, i])

    pca_features_rgb = pca_features.copy()
    pca_features_rgb[bg_mask] = 0
    pca_features_rgb[fg_mask] = pca_features_fg

    pca_features_rgb = pca_features_rgb.reshape(batch_size, h, w, 3)
    if not interpolation:
        H = W = 128
        scale = H // h
        interpolated_pca_features = np.zeros((batch_size, H, W, 3), dtype=pca_features_rgb.dtype)
        for i in range(len(pca_features_rgb)):
            for j in range(h):
                for k in range(w):
                    interpolated_pca_features[i, scale * j : scale * (j + 1), scale * k : scale * (k + 1)] = (
                        pca_features_rgb[i, j, k]
                    )
        pca_features_rgb = interpolated_pca_features
    else:
        pca_features_rgb = np.stack([cv2.resize(p, (128, 128)) for p in pca_features_rgb])
    return pca_features_rgb, pca

class Theia_DINOv2(nn.Module):
    """
    TheiaModelDINO 类，用于处理 RGB 图像并提取特征。

    输入:
        images: Tensor, 形状为 (batch_size, seq_len, C, H, W) 或 (batch_size, C, H, W)

    输出:
        fused_features: Tensor, 提取并融合后的特征
    """

    def __init__(self, theia_model_name: str, target_model_name: str, device: str):
        super(Theia_DINOv2, self).__init__()
        self.device = device

        # 加载预训练的 TheiaModel
        self.theia_model = AutoModel.from_pretrained(theia_model_name, trust_remote_code=True)
        self.theia_model.to(self.device)
        self.theia_model.eval()  # 设置为评估模式
        self.target_model_name = target_model_name

        self.feature_means, self.feature_vars = load_theia_feature_stats(self.target_model_name,
                                                         stat_file_root="/home/zqy/code/RL2/theia/feature_stats")

        # 定义一个简单的下采样层（可选）
        # self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # 获取特征输出 dino (bs,1024,16,16)
            features = self.theia_model(images)[self.target_model_name].detach().cpu()
            features = denormalize_feature(features, self.feature_means, self.feature_vars)
            # dino_gt_feature = rearrange(features, "b c h w -> b (h w) c")
        dino_gt_dec2, pca = decode_dinov2(features, pca=None)


        return dino_gt_dec2
