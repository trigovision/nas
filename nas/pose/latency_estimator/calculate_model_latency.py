import argparse

import torch
from torch import nn

from nas.imagenet_classification.elastic_nn.networks import OFAMobileNetV3
from nas.imagenet_classification.networks import MobileNetV3Large
from nas.pose.elastic_nn.ofa_mbv3 import POFAMobileNetV3
from nas.pose.latency_estimator.build_latency_table import calculate_layer_latency, IMAGE_SIZE
from nas.pose.networks.mobilenet_v3 import PMobileNetV3Large


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

    model = PMobileNetV3Large(
        width_mult=1.0,
        ks=7,
        expand_ratio=6,
        depth_param=4,
    )

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
