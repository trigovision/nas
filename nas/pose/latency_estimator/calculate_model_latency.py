import argparse

import torch
from pose.models.backbones.resnet50_full import resnet50_full
from torch import nn

from nas.imagenet_classification.elastic_nn.networks import OFAMobileNetV3, OFAResNets
from nas.imagenet_classification.networks import MobileNetV3Large, ResNet50
from nas.pose.elastic_nn.ofa_mbv3 import POFAMobileNetV3
from nas.pose.elastic_nn.ofa_resnet import POFAResNet
from nas.pose.latency_estimator.build_latency_table import calculate_layer_latency, IMAGE_SIZE
from nas.pose.networks.mobilenet_v3 import PMobileNetV3Large
from nas.pose.networks.resnet import PResNet50


class DummyBackbone(nn.Module):
    def forward(self, x):
        return x


NUM_BACKBONE_CHANNELS = 128


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--arch", type=str, choices=["mobilenetv3"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trt", action="store_true")
    parser.add_argument("--int8", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    # model = OFAResNets(
    #     depth_list=[0, 1, 2],
    #     expand_ratio_list=[0.2, 0.25, 0.35],
    #     width_mult_list=[0.65, 0.85, 1.0],
    # )
    # model = PResNet50(width_mult=1.0, expand_ratio=0.35, depth_param=2)
    model2 = resnet50_full()
    model = POFAResNet(expand_ratio_list=0.25, depth_list=0)
    import ipdb

    ipdb.set_trace()

    input_size = [args.batch_size, 3] + IMAGE_SIZE
    latency_info = calculate_layer_latency(
        model,
        input_size=input_size,
        trt=args.trt,
        int8=args.int8,
        output_names=["pafs", "pacs", "heatmaps"],
        silent=False,
    )
    print(latency_info)
