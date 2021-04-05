import torch.nn as nn
import torch.nn.functional as F

FE_STAGE0 = 64
FE_STAGE1 = 128
BN_IO = 256


def ConvReLU(n_in, n_out, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(n_out),
    )


class Bottleneck(nn.Module):
    def __init__(self, n_in, n_out, stride=1, downsample=None):
        super().__init__()
        assert 1 <= stride <= 2

        n_mid = n_out // 2
        self.bottleneck = nn.Sequential(
            ConvReLU(n_in, n_mid),
            ConvReLU(n_mid, n_mid, 3, stride, padding=1),
            ConvReLU(n_mid, n_out),
        )

        residual = []
        if n_in != n_out:
            residual.append(ConvReLU(n_in, n_out))
        if stride == 2:
            residual.append(nn.MaxPool2d(2, stride=2))
        self.residual = nn.Sequential(*residual)

    def forward(self, x):
        x0 = self.residual(x)
        x = self.bottleneck(x)
        return x0 + x


def ChainedBottleneck(n_io, num_blocks):
    layers = []
    for i in range(0, num_blocks):
        layers.append(Bottleneck(n_io, n_io))
    return nn.Sequential(*layers)


class Hourglass(nn.Module):
    def __init__(self, depth, num_blocks):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.hg = self._make_hour_glass(depth, num_blocks)

    def _make_hour_glass(self, depth, num_blocks):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3 + int(i == 0)):
                res.append(ChainedBottleneck(BN_IO, num_blocks))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        x0 = self.hg[n - 1][0](x)
        x = F.max_pool2d(x0, 2, stride=2)
        x = self.hg[n - 1][1](x)
        x = self._hour_glass_forward(n - 1, x) if n > 1 else self.hg[n - 1][3](x)
        x = self.hg[n - 1][2](x)
        x = F.interpolate(x, scale_factor=2)
        return x + x0

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self, depth=4, num_stacks=2, num_blocks=1, num_classes=16):
        super(HourglassNet, self).__init__()

        self.num_stacks = num_stacks
        self.feature_extractor = nn.Sequential(
            ConvReLU(3, FE_STAGE0, 7, stride=2, padding=3),
            Bottleneck(FE_STAGE0, FE_STAGE1, 1),
            nn.MaxPool2d(2, stride=2),
            Bottleneck(FE_STAGE1, FE_STAGE1, 1),
            Bottleneck(FE_STAGE1, BN_IO, 1),
        )

        hg, score = [], []
        for i in range(num_stacks):
            hg.append(
                nn.Sequential(
                    Hourglass(depth, num_blocks),
                    ChainedBottleneck(BN_IO, num_blocks),
                    ConvReLU(BN_IO, BN_IO),  # This layout prevent numerical issues
                )
            )
            score.append(nn.Conv2d(BN_IO, num_classes, kernel_size=1))

        self.hg = nn.ModuleList(hg)
        self.score = nn.ModuleList(score)

    def forward(self, x):
        x = self.feature_extractor(x)
        out = []
        for hg, score in zip(self.hg, self.score):
            y = hg(x)
            out.append(score(y))
            x = x + y
        return out
