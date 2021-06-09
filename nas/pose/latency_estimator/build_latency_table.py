import argparse
from time import time
from typing import List

import numpy as np
import tensorrt as trt
import torch
from torch import nn
from torch2trt import torch2trt
from tqdm import tqdm
from trigo_logging import logger

from nas.utils.pytorch_utils import no_grad

NUM_WARMUP_ITERS = 50
NUM_ACTUAL_ITERS = 50
IMAGE_SIZE = [360, 640]


def get_time():
    torch.cuda.synchronize()
    return time()


def get_enumerator(lst, silent):
    if silent:
        return lst
    else:
        return tqdm(lst)


def log(msg, silent):
    if not silent:
        logger.info(msg)


@no_grad
def calculate_layer_latency(
    layer: nn.Module, input_size: List[int], trt=False, int8=False, output_names=["output"], silent=True
):
    input = torch.rand(input_size, device="cuda:0").half()

    layer = layer.cuda().eval()
    layer = layer.half()

    if trt:
        log("Converting to trt...", silent)
        torch.cuda.empty_cache()
        layer = layer_to_trt(layer, input, output_names, int8_mode=int8, silent=silent)

    log("Warming up...", silent)
    for _ in get_enumerator(range(NUM_WARMUP_ITERS), silent):
        layer(input)

    log("Running net...", silent)
    times = []
    for _ in get_enumerator(range(NUM_ACTUAL_ITERS), silent):
        time_before = get_time()

        layer(input)

        times.append((get_time() - time_before) * 1000)

    times_arr = np.array(times)
    return {"std_ms": times_arr.std(), "mean_ms": times_arr.mean(), "mean_fps": input_size[0] * 1000 / times_arr.mean()}


def layer_to_trt(layer, input, output_names, int8_mode=False, silent=True):
    if silent:
        log_level = trt.Logger.ERROR
    else:
        log_level = trt.Logger.INFO

    return torch2trt(
        layer,
        inputs=[input],
        input_names=["input"],
        output_names=output_names,
        max_batch_size=input.shape[0],
        int8_mode=int8_mode,
        fp16_mode=True,
        log_level=log_level,
    )


def main(batch_size: int, output_path: str, tensorrt: bool, int8: bool):
    latency_estimator = LatencyEstimator()
    layer_cfgs = StageSearchSpace.get_layer_types()

    for layer_cfg in tqdm(layer_cfgs):
        layer = PoseStageBlock(**layer_cfg)

        input_size = [batch_size, layer_cfg["in_channels"]] + INPUT_SIZE
        latency_info = calculate_layer_latency(layer, input_size, trt=tensorrt, int8=int8)

        layer_cfg["input_size"] = input_size
        latency_estimator.add_key(layer_cfg, latency_info)

    latency_estimator.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("output_path", type=str)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trt", action="store_true")
    parser.add_argument("--int8", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    main(args.batch_size, args.output_path, args.trt, args.int8)
