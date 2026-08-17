from functools import partial

import torch.nn as nn


def number_of_features_per_level(init_channels: int, num_levels: int) -> list:
    return [init_channels * 2 ** k for k in range(num_levels)]


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding):
    assert "c" in order
    modules = []
    for i, char in enumerate(order):
        if char == "r":
            modules.append(("ReLU", nn.ReLU(inplace=True)))
        elif char == "c":
            bias = "g" not in order
            modules.append(("conv", nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)))
        elif char == "g":
            is_before_conv = i < order.index("c")
            num_channels = in_channels if is_before_conv else out_channels
            n_groups = 1 if num_channels < num_groups else num_groups
            modules.append(("groupnorm", nn.GroupNorm(num_groups=n_groups, num_channels=num_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'")
    return modules


class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, order="gcr", num_groups=8, padding=1):
        super().__init__()
        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order="gcr",
                 num_groups=8, padding=1, upscale=2):
        super().__init__()
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels if upscale == 1 else out_channels // 2
            conv1_out_channels = max(conv1_out_channels, in_channels)
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.add_module(
            "SingleConv1",
            SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups, padding),
        )
        self.add_module(
            "SingleConv2",
            SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups, padding),
        )


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, apply_pooling=True, pool_kernel_size=2,
                 conv_kernel_size=3, conv_layer_order="gcr", num_groups=8, padding=1, upscale=2):
        super().__init__()
        self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size) if apply_pooling else None
        self.basic_module = DoubleConv(
            in_channels, out_channels, encoder=True, kernel_size=conv_kernel_size,
            order=conv_layer_order, num_groups=num_groups, padding=padding, upscale=upscale,
        )

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        return self.basic_module(x)


def create_encoders(in_channels, f_maps, conv_kernel_size, conv_padding, conv_upscale,
                     layer_order, num_groups, pool_kernel_size):
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(
                in_channels, out_feature_num, apply_pooling=False,
                conv_layer_order=layer_order, conv_kernel_size=conv_kernel_size,
                num_groups=num_groups, padding=conv_padding, upscale=conv_upscale,
            )
        else:
            encoder = Encoder(
                f_maps[i - 1], out_feature_num,
                conv_layer_order=layer_order, conv_kernel_size=conv_kernel_size,
                num_groups=num_groups, pool_kernel_size=pool_kernel_size,
                padding=conv_padding, upscale=conv_upscale,
            )
        encoders.append(encoder)
    return nn.ModuleList(encoders)


class VoxelFeatureEncoder(nn.Module):
    def __init__(self, in_channels=1, f_maps=(32, 64, 128, 256), layer_order="gcr",
                 num_groups=8, conv_kernel_size=3, conv_padding=1, conv_upscale=2, pool_kernel_size=2):
        super().__init__()
        f_maps = list(f_maps)
        self.encoders = create_encoders(
            in_channels, f_maps, conv_kernel_size, conv_padding, conv_upscale,
            layer_order, num_groups, pool_kernel_size,
        )
        self.feature_dims = f_maps

    def forward(self, x):
        features = []
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)
        return features
