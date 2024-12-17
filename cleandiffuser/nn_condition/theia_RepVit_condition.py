import os
from typing import Dict, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.models.layers import SqueezeExcite
from transformers import AutoModel

from cleandiffuser.nn_condition import BaseNNCondition

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

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


class RepVitObsCondition(BaseNNCondition):
    """
    Input:
        - condition: {"cond1": (b, *cond1_shape), "cond2": (b, *cond2_shape), ...} or (b, *cond_in_shape)
        - mask :     (b, *mask_shape) or None, None means no mask

    Output:
        - condition: (b, *cond_out_shape)

    Assumes rgb input: B, C, H, W or B, seq_len, C,H,W
    Assumes low_dim input: B, D or B, seq_len, D
    """

    def __init__(self,
                 shape_meta: dict,
                 rgb_model_name: str,
                 emb_dim: int = 256,
                 resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
                 share_rgb_model: bool = False,
                 # use_seq: B, seq_len, C, H, W or B, C, H, W
                 image_channel: int = 16,
                 use_seq=False,
                 # if True: (bs, seq_len, embed_dim)
                 keep_horizon_dims=False
                 ):
        super().__init__()
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        # rgb_model
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
        rgb_model = RepViT(cfgs, image_channel = image_channel, num_classes=emb_dim, distillation=False)


        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # print(key, attr)
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h, w)
                    )
                    input_shape = (h, w,shape[2])

                # configure randomizer
                this_randomizer = nn.Identity()
                # configure normalizer
                this_normalizer = nn.Identity()

                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        self.use_seq = use_seq
        self.keep_horizon_dims = keep_horizon_dims
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.output_shape(), emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, emb_dim))
        self.mlp = nn.Sequential(
            nn.Linear(self.output_shape(), emb_dim))

    def multi_image_forward(self, obs_dict):
        batch_size = None
        features = list()

        if self.use_seq:
            # input: (bs, horizon, h, w, c)
            for k in obs_dict.keys():
                obs_dict[k] = obs_dict[k].flatten(end_dim=1)

        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            # imgs = list()
            # for key in self.rgb_keys:
            #     img = obs_dict[key]
            #     if batch_size is None:
            #         batch_size = img.shape[0]
            #     else:
            #         assert batch_size == img.shape[0]
            #     # print( img.shape[1:] )
            #     # print(self.key_shape_map[key])
            #     assert img.shape[1:] == self.key_shape_map[key]
            #     ## img = self.key_transform_map[key](img)
            #     imgs.append(img)
            # # (N*B,H,W,C)
            # imgs = torch.cat(imgs, dim=0)
            # # (N*B,D)
            # feature = self.key_model_map['rgb'](imgs)
            # # (N,B,D)
            # feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            # # (B,N,D)
            # feature = torch.moveaxis(feature, 0, 1)
            # # (B,N*D)
            # feature = feature.reshape(batch_size, -1)
            # features.append(feature)
            # 处理所有 RGB 输入
            imgs = torch.cat([obs_dict[key] for key in self.rgb_keys],
                             dim=0)  # (N*B, H, W, C)
            batch_size = obs_dict[self.rgb_keys[0]].shape[0]  # 获取批量大小
            feature = self.key_model_map['rgb'](imgs)  # (N*B, D)
            feature = feature.view(len(self.rgb_keys), batch_size, -1).permute(1, 0, 2).reshape(batch_size,-1)  # (B, N*D)
            features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        # concatenate all features
        features = torch.cat(features, dim=-1)
        return features

    def forward(self, obs_dict, mask=None):
        ori_batch_size, ori_seq_len = self.get_batch_size(obs_dict)
        features = self.multi_image_forward(obs_dict)
        # linear embedding
        result = self.mlp(features)
        if self.use_seq:
            if self.keep_horizon_dims:
                result = result.reshape(ori_batch_size, ori_seq_len, -1)
            else:
                result = result.reshape(ori_batch_size, -1)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 8
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            if self.use_seq:
                prefix = (batch_size, 1)
            else:
                prefix = (batch_size,)
            this_obs = torch.zeros(
                prefix + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.multi_image_forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape[0]

    def get_batch_size(self, obs_dict):
        any_key = next(iter(obs_dict))
        any_tensor = obs_dict[any_key]
        return any_tensor.size(0), any_tensor.size(1)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

class TheiaModelSAM(nn.Module):
    def __init__(self, theia_model_name, target_model_names, use_seq, device):
        super(TheiaModelSAM, self).__init__()
        self.device = device
        self.theia_model = AutoModel.from_pretrained(theia_model_name, trust_remote_code=True).to(self.device)
        self.target_model_names = target_model_names
        self.use_seq = use_seq


    def process_images(self, images):
        feature_input = []
        with torch.no_grad():
            batch_features = self.theia_model(images)
            for i in self.target_model_names:
                # print(i, batch_features[i].shape)
                feature_input.append(batch_features[i])
        return feature_input

    def fuse_features(self, feature_input):
        batch_size = feature_input[0].size(0)
        feature_input_SAM = feature_input[0].view(batch_size, 16, 256, 256)
        tensor_downsampled = F.max_pool2d(feature_input_SAM, kernel_size=2, stride=2)
        return tensor_downsampled

    def forward(self, images):
        bs_ori = images.size(0)
        # print(bs_ori)
        if self.use_seq:
            # batch_size = images.size(0)
            images = images.reshape(bs_ori * images.size(1), *images.shape[2:])
            # images = torch.cat(images, dim=0)
        feature_inputs = self.process_images(images)
        # feature_inputs = [features[name] for name in self.target_model_names]
        fused_features = self.fuse_features(feature_inputs)
        if self.use_seq:
            # print(fused_features.shape)
            fused_features = fused_features.reshape(bs_ori, -1, *fused_features.shape[1:])
        # print(fused_features.shape)
        # output = self.conv_model(fused_features)
        return fused_features

class TheiaModelDINO(nn.Module):
    def __init__(self, theia_model_name, target_model_names, use_seq, device):
        super(TheiaModelDINO, self).__init__()
        self.device = device
        self.theia_model = AutoModel.from_pretrained(theia_model_name, trust_remote_code=True).to(self.device)
        self.target_model_names = target_model_names
        self.use_seq = use_seq


    def process_images(self, images):
        feature_input = []
        with torch.no_grad():
            batch_features = self.theia_model(images)
            for i in self.target_model_names:
                # print(i, batch_features[i].shape)
                feature_input.append(batch_features[i])
        return feature_input

    def fuse_features(self, feature_input):
        batch_size = feature_input[0].size(0)
        feature_input_SAM = feature_input[0].view(batch_size, 4, 256, 256)
        tensor_downsampled = F.max_pool2d(feature_input_SAM, kernel_size=2, stride=2)
        return tensor_downsampled

    def forward(self, images):
        bs_ori = images.size(0)
        # print(bs_ori)
        if self.use_seq:
            # batch_size = images.size(0)
            images = images.reshape(bs_ori * images.size(1), *images.shape[2:])
            # images = torch.cat(images, dim=0)
        feature_inputs = self.process_images(images)
        # feature_inputs = [features[name] for name in self.target_model_names]
        fused_features = self.fuse_features(feature_inputs)
        if self.use_seq:
            # print(fused_features.shape)
            fused_features = fused_features.reshape(bs_ori, -1, *fused_features.shape[1:])
        # print(fused_features.shape)
        # output = self.conv_model(fused_features)
        return fused_features
if __name__ == "__main__":
    shape_meta = {
        'obs': {
            'image': {
                'shape': (4, 128, 128),
                'type': 'rgb'
            },
            'agent_pos': {
                'shape': (31,),
                'type': 'low_dim'
            }
        }
    }
    rgb_model = "RepVit"

    resize_shape = None
    # crop_shape=(84, 84)
    # random_crop=True
    # use_group_norm=True
    share_rgb_model = True
    # imagenet_norm=False
    # im = TheiaImageObsCondition(shape_meta, emb_dim=256, rgb_model_name=rgb_model, resize_shape=resize_shape,
    #                             share_rgb_model=share_rgb_model)
    im = RepVitObsCondition(shape_meta, emb_dim=256, rgb_model_name=rgb_model, resize_shape=resize_shape,
                                share_rgb_model=share_rgb_model, use_seq = False)
    print(im.output_shape())

    # 创建虚拟输入数据
    dummy_obs = {
        # 'image': torch.randint(0, 1, (16, 8, 16, 128, 128), dtype=torch.uint8),  # 模拟点云数据
        'image': torch.randn(8, 16, 128, 128),  # 模拟点云数据
        'agent_pos': torch.randn(8, 31)  # 模拟低维度数据
    }

    # 设置模型为评估模式并检查输出
    im.eval()
    with torch.no_grad():
        eval_output = im(dummy_obs)
        print("Evaluation Output:", eval_output.shape)

    # 设置模型为训练模式并检查输出
    im.train()
    train_output = im(dummy_obs)
    print("Training Output:", train_output.shape)

