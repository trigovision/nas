from os.path import join

import cv2

from pose.train.train_args import get_train_parser, process_train_args
from pose.train.trainer import Trainer
import torch
from torch.nn.parallel import DistributedDataParallel

from nas.pose.elastic_nn.ofa_mbv3 import POFAMobileNetV3
from nas.pose.train.load_state import load_state

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader


class OFAPoseTrainer(Trainer):
    """
    This class overloads some of pose's Trainer's methods so that we train
    for OFA
    """

    def _load_state(self):
        self.current_iter = 0
        self.current_epoch = 0
        self.start_iter = 0

        assert self.args.checkpoint_path is not None
        assert self.args.weights_mode != "resume", f"Resume isn't supported for now!"

        checkpoint = torch.load(
            self.args.checkpoint_path, map_location=lambda storage, loc: storage.cuda(self.args.device_id)
        )

        load_state(self.net, checkpoint)

    def _init_model(self):
        self.args.width_mult_list = 1.2

        if self.args.phase == "full":
            self.args.ks_list = [7]
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
        self.net = self.net.cuda()
        self.net.train()

        if self.args.distributed:
            self.model = DistributedDataParallel(
                self.net, device_ids=[self.args.device_id], output_device=self.args.device_id
            )
        else:
            self.model = self.net


if __name__ == "__main__":
    parser = get_train_parser()
    parser.add_argument(
        "--phase", choices=["full", "kernel", "depth", "expand"], type=str, required=True, help="training phase"
    )

    args = parser.parse_args()
    args = process_train_args(args)

    trainer = OFAPoseTrainer(args)
    trainer.fit()
