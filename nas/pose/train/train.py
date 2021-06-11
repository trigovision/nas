import random

import cv2
import torch
from pose.train.train_args import get_train_parser, process_train_args
from pose.train.trainer import Trainer
from torch.nn.parallel import DistributedDataParallel

from nas.pose.elastic_nn.ofa_mbv3 import POFAMobileNetV3
from nas.pose.elastic_nn.ofa_resnet import POFAResNet
from nas.pose.train.load_state import load_state
from nas.utils import subset_mean

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader


class OFAPoseTrainer(Trainer):
    """
    This class overloads some of pose's Trainer's methods so that we train
    for OFA
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_iters = len(self.train_loader)

    # def run_batch(self, model: torch.nn.Module, batch: dict, log_scalars: dict, log_metrics: bool = True) -> tuple:
    #     import ipdb
    #
    #     ipdb.set_trace()
    #
    #     # This logic was taken from progressive_shrinking.py
    #     subnet_str = ""
    #     loss_of_subnets = []
    #
    #     for dynamic_batch_idx in range(self.args.dynamic_batch_size):
    #         subnet_seed = int(
    #             "%d%.3d%.3d" % (self.current_epoch * self.num_iters + self.current_iter, dynamic_batch_idx, 0)
    #         )
    #         random.seed(subnet_seed)
    #         subnet_settings = model.sample_active_subnet()
    #         subnet_str += (
    #             "%d: " % dynamic_batch_idx
    #             + ",".join(
    #                 [
    #                     "%s_%s"
    #                     % (
    #                         key,
    #                         "%.1f" % subset_mean(val, 0) if isinstance(val, list) else val,
    #                     )
    #                     for key, val in subnet_settings.items()
    #                 ]
    #             )
    #             + " || "
    #         )
    #
    #         losses_dict, outputs_dict = super().run_batch(model, batch, log_scalars, log_metrics)
    #         loss_of_subnets.append(losses_dict)
    #
    #     # We return a list of all the subnets' losses, this will be handled in _aggregate_batch_losses overload
    #     # Regarding the outputs, we just arbitrarily return the last dict so that it can be logged..
    #     return loss_of_subnets, outputs_dict
    #
    # def _aggregate_batch_losses(self, subnet_losses):
    #     import ipdb
    #
    #     ipdb.set_trace()
    #     subnet_losses = []
    #     for losses_dict in subnet_losses:
    #         subnet_losses.append(super()._aggregate_batch_losses(losses_dict))
    #     return torch.stack(subnet_losses).mean()

    def _load_state(self):
        self.current_iter = 0
        self.current_epoch = 0
        self.start_iter = 0

        assert self.args.checkpoint_path is not None
        assert self.args.weights_mode != "resume", f"Resume isn't supported for now!"

        checkpoint = torch.load(
            self.args.checkpoint_path, map_location=lambda storage, loc: storage.cuda(self.args.device_id)
        )

        load_state(self.net, checkpoint, self.args.arch)

    def _init_model(self):
        if self.args.arch == "mbv3":
            self._init_mbv3_model()
        elif self.args.arch == "resnet":
            self._init_resnet_model()

        self.net = self.net.cuda()
        self.net.train()

        if self.args.distributed:
            self.model = DistributedDataParallel(
                self.net, device_ids=[self.args.device_id], output_device=self.args.device_id
            )
        else:
            self.model = self.net

    def _init_mbv3_model(self):
        self.args.width_mult_list = 1.2

        if self.args.phase == "full":
            args.dynamic_batch_size = 1
            self.args.ks_list = [7]
            self.args.expand_list = [6]
            self.args.depth_list = [4]
        elif self.args.phase == "depth":
            args.dynamic_batch_size = 1
            self.args.ks_list = [3, 5, 7]
            self.args.expand_list = [6]
            self.args.depth_list = [4]
        else:
            raise NotImplementedError()

        self.net = POFAMobileNetV3(
            width_mult=self.args.width_mult_list,
            ks_list=self.args.ks_list,
            expand_ratio_list=self.args.expand_list,
            depth_list=self.args.depth_list,
        )

    def _init_resnet_model(self):
        if self.args.phase == "full":
            self.args.expand_list = [0.35]
            self.args.depth_list = [2]
            self.args.width_mult_list = [1]
            self.args.dynamic_batch_size = 1
        elif self.args.phase == "depth":
            self.args.expand_list = [0.35]
            self.args.depth_list = [0, 1, 2]
            self.args.width_mult_list = [1]
            self.args.dynamic_batch_size = 2
        else:
            raise NotImplementedError()

        self.net = POFAResNet(
            width_mult_list=self.args.width_mult_list,
            expand_ratio_list=self.args.expand_list,
            depth_list=self.args.depth_list,
        )


if __name__ == "__main__":
    parser = get_train_parser()
    parser.add_argument(
        "--phase", choices=["full", "depth", "expand", "width"], type=str, required=True, help="training phase"
    )
    parser.add_argument("--arch", choices=["mbv3", "resnet"], type=str, required=True, help="OFA architecture")

    args = parser.parse_args()
    args = process_train_args(args)

    trainer = OFAPoseTrainer(args)
    trainer.fit()
