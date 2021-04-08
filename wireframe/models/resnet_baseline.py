import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=has_bias
    )


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class ResNetUperNet(nn.Module):
    def __init__(self, net_enc, net_dec):
        super(ResNetUperNet, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.num_stacks = 1

    def forward(self, x):
        pred = [self.decoder(self.encoder(x, output_feat_pyramid=True))]
        return pred


class ResNet(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResNet, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, output_feat_pyramid=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if output_feat_pyramid:
            return conv_out
        return [x]


class SubHead(nn.Module):
    def __init__(self, input_channels):
        super(SubHead, self).__init__()
        output_channels = int(input_channels / 4)
        self.jc_cls_feat = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.jc_cls_head = nn.Conv2d(output_channels, 2, kernel_size=1)
        self.jd_cls_feat = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.jd_cls_head = nn.Conv2d(output_channels, 2, kernel_size=1)
        self.line_reg_feat = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.line_reg_head = nn.Conv2d(output_channels, 1, kernel_size=1)
        self.j_reg_feat = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.j_reg_head = nn.Conv2d(output_channels, 4, kernel_size=1)
        self.ltheta_reg_feat = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.ltheta_reg_head = nn.Conv2d(output_channels, 1, kernel_size=1)
        self.depth_reg_feat = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.depth_reg_head = nn.Conv2d(output_channels, 3, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        jc_cls_head = self.jc_cls_head(self.relu(self.jc_cls_feat(x)))
        jd_cls_head = self.jd_cls_head(self.relu(self.jd_cls_feat(x)))
        line_reg_head = self.line_reg_head(self.relu(self.line_reg_feat(x)))
        j_reg_head = self.j_reg_head(self.relu(self.j_reg_feat(x)))
        ltheta_reg_head = self.ltheta_reg_head(self.relu(self.ltheta_reg_feat(x)))
        depth_reg_head = self.depth_reg_head(self.relu(self.depth_reg_feat(x)))
        out = torch.cat([jc_cls_head, jd_cls_head, line_reg_head,
                         j_reg_head, ltheta_reg_head, depth_reg_head], 1)
        return out
    
    
class UPerNet(nn.Module):
    def __init__(
        self,
        num_class=150,
        fc_dim=4096,
        pool_scales=(1, 2, 3, 6),
        fpn_inplanes=(256, 512, 1024, 2048),
        fpn_dim=256,
    ):
        super(UPerNet, self).__init__()

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(
                nn.Sequential(
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
            )
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(
            fc_dim + len(pool_scales) * 512, fpn_dim, 1
        )

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:  # skip the top layer
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(fpn_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            # nn.Conv2d(fpn_dim, num_class, kernel_size=1),
        )
        self.conv_last_last = SubHead(fpn_dim)

    def forward(self, conv_out):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(
                pool_conv(
                    nn.functional.interpolate(
                        pool_scale(conv5),
                        (input_size[2], input_size[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            )
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode="bilinear", align_corners=False
            )  # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(
                nn.functional.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        x = self.conv_last_last(x)
        return x
