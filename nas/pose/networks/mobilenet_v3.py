# This file was copied from imagenet_classification.mobilenet_v3 and altered as follows:
# 1. Replace stride with dilation for the last 2 layers
# 2. Run pose heads at the end of the network

import copy

import torch.nn as nn

from nas.pose.networks.pose import PoseHeads
from nas.utils import make_divisible, MyNetwork
from nas.utils.layers import ConvLayer, IdentityLayer, MBConvLayer, ResidualBlock, set_layer_from_config


class PMobileNetV3(MyNetwork):
    def __init__(self, first_conv, blocks, final_expand_layer, num_kps=7, num_conn_types=15):
        super(PMobileNetV3, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer

        self.head = PoseHeads(backbone_channels=self.num_channels, num_kps=num_kps, num_conn_types=num_conn_types)

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        features = self.final_expand_layer(x)

        return self.head(features)

    @property
    def module_str(self):
        _str = self.first_conv.module_str + "\n"
        for block in self.blocks:
            _str += block.module_str + "\n"
        _str += self.final_expand_layer.module_str + "\n"
        return _str

    @property
    def config(self):
        return {
            "name": PMobileNetV3.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
            "final_expand_layer": self.final_expand_layer.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config["first_conv"])
        final_expand_layer = set_layer_from_config(config["final_expand_layer"])

        blocks = []
        for block_config in config["blocks"]:
            blocks.append(ResidualBlock.build_from_config(block_config))

        net = PMobileNetV3(
            first_conv,
            blocks,
            final_expand_layer,
        )
        if "bn" in config:
            net.set_bn_param(**config["bn"])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-5)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                if isinstance(m.conv, MBConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.conv.point_linear.bn.weight.data.zero_()

    @property
    def grouped_block_index(self):
        info_list = []
        block_index_list = []
        for i, block in enumerate(self.blocks[1:], 1):
            if block.shortcut is None and len(block_index_list) > 0:
                info_list.append(block_index_list)
                block_index_list = []
            block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list

    @staticmethod
    def build_net_via_cfg(cfg, input_channel, last_channel):
        # first conv layer
        first_conv = ConvLayer(
            3,
            input_channel,
            kernel_size=3,
            stride=2,
            use_bn=True,
            act_func="h_swish",
            ops_order="weight_bn_act",
        )
        # build mobile blocks
        feature_dim = input_channel
        blocks = []
        for stage_id, block_config_list in cfg.items():
            for (k, mid_channel, out_channel, use_se, act_func, stride, expand_ratio, dilation) in block_config_list:
                mb_conv = MBConvLayer(
                    feature_dim, out_channel, k, stride, expand_ratio, mid_channel, act_func, use_se, dilation=dilation
                )
                if stride == 1 and out_channel == feature_dim:
                    shortcut = IdentityLayer(out_channel, out_channel)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mb_conv, shortcut))
                feature_dim = out_channel
        # final expand layer
        final_expand_layer = ConvLayer(
            feature_dim,
            last_channel,
            kernel_size=1,
            use_bn=True,
            act_func="h_swish",
            ops_order="weight_bn_act",
        )

        return first_conv, blocks, final_expand_layer

    @staticmethod
    def adjust_cfg(cfg, ks=None, expand_ratio=None, depth_param=None, stage_width_list=None):
        for i, (stage_id, block_config_list) in enumerate(cfg.items()):
            for block_config in block_config_list:
                if ks is not None and stage_id != "0":
                    block_config[0] = ks
                if expand_ratio is not None and stage_id != "0":
                    block_config[-2] = expand_ratio
                    block_config[1] = None
                    if stage_width_list is not None:
                        block_config[2] = stage_width_list[i]
            if depth_param is not None and stage_id != "0":
                new_block_config_list = [block_config_list[0]]
                new_block_config_list += [copy.deepcopy(block_config_list[-1]) for _ in range(depth_param - 1)]
                cfg[stage_id] = new_block_config_list
        return cfg

    def load_state_dict(self, state_dict, **kwargs):
        current_state_dict = self.state_dict()

        for key in state_dict:
            if key not in current_state_dict:
                assert ".mobile_inverted_conv." in key
                new_key = key.replace(".mobile_inverted_conv.", ".conv.")
            else:
                new_key = key
            current_state_dict[new_key] = state_dict[key]
        super(PMobileNetV3, self).load_state_dict(current_state_dict)


class PMobileNetV3Large(PMobileNetV3):
    def __init__(
        self,
        width_mult=1.0,
        bn_param=(0.1, 1e-5),
        ks=None,
        expand_ratio=None,
        depth_param=None,
        stage_width_list=None,
    ):
        input_channel = 16
        self.num_channels = 1280

        input_channel = make_divisible(input_channel * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
        self.num_channels = (
            make_divisible(self.num_channels * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            if width_mult > 1.0
            else self.num_channels
        )

        cfg = {
            # kernel, mid_channel, out_channel, use_se, act_func, stride, expand_ratio, dilation
            "0": [
                [3, 16, 16, False, "relu", 1, 1, 1],
            ],
            "1": [
                [3, 64, 24, False, "relu", 2, None, 1],  # 4
                [3, 72, 24, False, "relu", 1, None, 1],  # 3
            ],
            "2": [
                [5, 72, 40, True, "relu", 2, None, 1],  # 3
                [5, 120, 40, True, "relu", 1, None, 1],  # 3
                [5, 120, 40, True, "relu", 1, None, 1],  # 3
            ],
            "3": [
                [3, 240, 80, False, "h_swish", 1, None, 2],  # 6
                [3, 200, 80, False, "h_swish", 1, None, 1],  # 2.5
                [3, 184, 80, False, "h_swish", 1, None, 1],  # 2.3
                [3, 184, 80, False, "h_swish", 1, None, 1],  # 2.3
            ],
            "4": [
                [3, 480, 112, True, "h_swish", 1, None, 1],  # 6
                [3, 672, 112, True, "h_swish", 1, None, 1],  # 6
            ],
            "5": [
                [5, 672, 160, True, "h_swish", 1, None, 2],  # 6
                [5, 960, 160, True, "h_swish", 1, None, 1],  # 6
                [5, 960, 160, True, "h_swish", 1, None, 1],  # 6
            ],
        }

        cfg = self.adjust_cfg(cfg, ks, expand_ratio, depth_param, stage_width_list)
        # width multiplier on mobile setting, change `exp: 1` and `c: 2`
        for stage_id, block_config_list in cfg.items():
            for block_config in block_config_list:
                if block_config[1] is not None:
                    block_config[1] = make_divisible(block_config[1] * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
                block_config[2] = make_divisible(block_config[2] * width_mult, MyNetwork.CHANNEL_DIVISIBLE)

        (first_conv, blocks, final_expand_layer) = self.build_net_via_cfg(cfg, input_channel, self.num_channels)
        super(PMobileNetV3Large, self).__init__(first_conv, blocks, final_expand_layer)
        # set bn param
        self.set_bn_param(*bn_param)
