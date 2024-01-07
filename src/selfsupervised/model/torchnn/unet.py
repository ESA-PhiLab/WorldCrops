
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyparsing import Any


class depthwise_separable_conv(nn.Module):

    def __init__(self, nin, kernels_per_layer, nout) -> None:
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin,
                                   nin * kernels_per_layer,
                                   kernel_size=3,
                                   padding='same',
                                   groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer,
                                   nout,
                                   kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvDown(nn.Module):

    def __init__(self, input_dim, output_dim) -> None:
        super(ConvDown, self).__init__()

        self.conv_block = nn.Sequential(
            depthwise_separable_conv(input_dim, 1, output_dim), nn.ReLU(),
            nn.Dropout(p=0.3),
            depthwise_separable_conv(output_dim, 1, output_dim), nn.ReLU())

    def forward(self, x):
        return self.conv_block(x)


class ResidualDown(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ResidualDown, self).__init__()

        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2), )

    def forward(self, x):
        xpad = F.pad(x, [0, 1, 0, 1], mode='replicate')
        return self.conv_skip(xpad)


class ResidualBridge(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ResidualBridge, self).__init__()

        self.conv_block = nn.Sequential(
            depthwise_separable_conv(input_dim, 1, output_dim), nn.ReLU(),
            nn.Dropout(p=0.3),
            depthwise_separable_conv(output_dim, 1, output_dim), nn.ReLU())

    def forward(self, x):
        return self.conv_block(x)


class ResidualUp(nn.Module):

    def __init__(self, input_dim, output_dim) -> None:
        super(ResidualUp, self).__init__()

        self.conv_skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(input_dim, output_dim, kernel_size=1, padding='same'))

    def forward(self, x):
        return self.conv_skip(x)


class ConvUp(nn.Module):

    def __init__(self, input_dim, output_dim) -> None:
        super(ConvUp, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(input_dim + input_dim // 2,
                      output_dim,
                      kernel_size=3,
                      padding='same'), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding='same'),
            nn.ReLU(), nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, x):
        return self.conv_block(x)


class ResUnetEncoder(nn.Module):
    """ ResNet based Unet encoder """

    def __init__(self, channel=1, dropout=0.3, filters=None) -> None:
        super(ResUnetEncoder, self).__init__()
        args = filters
        self.pool = nn.MaxPool2d(2)
        self.down1 = ConvDown(args[0], args[1])
        self.res1 = ResidualDown(args[0], args[1])
        self.down2 = ConvDown(args[1], args[2])
        self.res2 = ResidualDown(args[1], args[2])
        self.bridge = ResidualBridge(args[2], args[3])

        # first layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, args[0], kernel_size=3, padding='same'),
            nn.ReLU(), nn.Dropout(p=dropout),
            depthwise_separable_conv(args[0], 1, args[0]), nn.ReLU())

        self.input_skip = nn.Sequential(
            nn.Conv2d(
                in_channels=channel,
                out_channels=args[0],
                kernel_size=3,
                stride=2,
            ))

    def forward(self, x) -> list[Any]:
        xpad = F.pad(x, [0, 1, 0, 1], mode='replicate')
        x1 = self.input_layer(x)
        x1b = self.pool(x1) + self.input_skip(xpad)
        x2 = self.down1(x1b)
        x2b = self.pool(x2) + self.res1(x1b)
        x3 = self.down2(x2b)
        x3b = self.pool(x3) + self.res2(x2b)
        x4 = self.bridge(x3b)

        embeddings = [x1, x2, x3, x3b, x4]
        return embeddings


class ResUnetDecoder(nn.Module):
    """ ResNet based Unet decoder """

    def __init__(self, dropout=0.3, filters=None) -> None:
        super(ResUnetDecoder, self).__init__()
        args = filters
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.res_bridge = ResidualUp(args[2], args[3])
        self.up1 = ConvUp(args[3], args[2])
        self.res3 = ResidualUp(args[3], args[2])
        self.up2 = ConvUp(args[2], args[1])
        self.res4 = ResidualUp(args[2], args[1])

        # last layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(args[1] + args[1] // 2,
                      args[0],
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(args[0], args[0], kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(args[0], 1, kernel_size=1, padding='same'),
            # nn.Sigmoid(), # done by BCEWithLogitsLoss
        )

    def forward(self, x1, x2, x3, x3b, x4):

        x4b = self.upsample(x4) + self.res_bridge(x3b)
        x5 = torch.cat([x4b, x3], dim=1)
        x6 = self.up1(x5) + self.res3(x4b)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up2(x7) + self.res4(x6)
        x9 = torch.cat([x8, x1], dim=1)
        output = self.output_layer(x9)

        return output


class ResUnet(nn.Module):
    """ ResNet based Unet with encoder, bridge and decoder """

    def __init__(self, channel, filters=[32, 64, 128, 256], dropout=0.3) -> None:
        super(ResUnet, self).__init__()

        # first layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding='same'),
            nn.ReLU(), nn.Dropout(p=dropout),
            depthwise_separable_conv(filters[0], 1, filters[0]), nn.ReLU())

        self.input_skip = nn.Sequential(
            nn.Conv2d(
                channel,
                filters[0],
                kernel_size=3,
                stride=2,
            ))
        self.pool = nn.MaxPool2d(2)
        self.down1 = ConvDown(filters[0], filters[1])
        self.res1 = ResidualDown(filters[0], filters[1])
        self.down2 = ConvDown(filters[1], filters[2])
        self.res2 = ResidualDown(filters[1], filters[2])

        self.bridge = ResidualBridge(filters[2], filters[3])

        self.up1 = ConvUp(filters[3], filters[2])
        self.res3 = ResidualUp(filters[3], filters[2])
        self.up2 = ConvUp(filters[2], filters[1])
        self.res4 = ResidualUp(filters[2], filters[1])

        # last layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[1] + filters[1] // 2,
                      filters[0],
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(filters[0], 1, kernel_size=1, padding='same'),
            # nn.Sigmoid(), # done by BCEWithLogitsLoss
        )

    def forward(self, x):
        xpad = F.pad(x, [0, 1, 0, 1], mode='replicate')
        x1 = self.input_layer(x)
        x1b = self.pool(x1) + self.input_skip(xpad)
        x2 = self.down1(x1b)
        x2b = self.pool(x2) + self.res1(x1b)
        x3 = self.down2(x2b)
        x3b = self.pool(x3) + self.res2(x2b)

        x4 = self.bridge(x3b)

        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up1(x5) + self.res3(x4)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up2(x7) + self.res4(x6)
        x9 = torch.cat([x8, x1], dim=1)
        output = self.output_layer(x9)

        return output
