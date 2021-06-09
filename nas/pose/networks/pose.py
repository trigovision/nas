import numbers

import torch
from torch import nn

from nas.utils.layers import ConvLayer

PAC_EMBEDDING_DIM = 32
GAUSSIAN_SMOOTHING_SIGMA = 0.875
GAUSSIAN_SMOOTHING_KERNEL_SIZE = 3


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2, padding=None):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            d2 = (mgrid - mean) ** 2
            kernel *= torch.exp(-d2 / 2 / std / std)
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        self.groups = channels
        if dim == 1:
            conv_class = torch.nn.Conv1d
        elif dim == 2:
            conv_class = torch.nn.Conv2d
        elif dim == 3:
            conv_class = torch.nn.Conv3d
        else:
            raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))
        self.conv = conv_class(channels, channels, kernel.shape, bias=False, padding=padding, groups=self.groups)
        for param in self.conv.parameters():
            param.requires_grad = False
        self.conv.weight.data.copy_(kernel)

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input)


class PoseHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=1,
            use_bn=False,
            bias=True,
            act_func="relu",
        )

    def forward(self, x):
        return self.conv(x)


class PACHead(nn.Module):
    def __init__(self, in_channels: int, num_conn_types: int):
        super().__init__()

        self.gaussian_smoothing = GaussianSmoothing(
            channels=PAC_EMBEDDING_DIM * num_conn_types * 2,
            kernel_size=GAUSSIAN_SMOOTHING_KERNEL_SIZE,
            sigma=GAUSSIAN_SMOOTHING_SIGMA,
            padding=1,
        )

        self.head = PoseHead(in_channels, PAC_EMBEDDING_DIM * num_conn_types * 2)

    def forward(self, x):
        pacs = self.head(x)
        return self.gaussian_smoothing(pacs)


class PAFHead(nn.Module):
    def __init__(self, in_channels: int, num_conn_types: int):
        super().__init__()

        self.head = PoseHead(in_channels, num_conn_types * 2)

    def forward(self, x):
        return self.head(x)


class KPHead(nn.Module):
    def __init__(self, in_channels: int, num_kps: int):
        super().__init__()

        self.head = PoseHead(in_channels, num_kps + 1)

    def forward(self, x):
        return self.head(x)


class PoseHeads(nn.Module):
    def __init__(self, backbone_channels: int, num_kps: int, num_conn_types: int):
        super().__init__()

        self.paf_head = PAFHead(backbone_channels, num_conn_types)
        self.pac_head = PACHead(backbone_channels, num_conn_types)
        self.kp_head = KPHead(backbone_channels + num_conn_types * 2, num_kps)

    def forward(self, x):
        pafs = self.paf_head(x)
        pacs = self.pac_head(x)
        heatmaps = self.kp_head(torch.cat([x, pafs], dim=1))

        return pafs, pacs, heatmaps
