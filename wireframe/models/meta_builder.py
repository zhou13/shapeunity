# initial copied from CSAIL semantic-segmentation-pytorch
# modified by Haozhi Qi
import torch
import torch.nn as nn

from wireframe.models.resnet_baseline import ResNet, UPerNet
from wireframe.models.backbones.resnet import get_resnet50


class MetaBuilder:
    # customized weights
    def init_weights(self, m):
        class_name = m.__class__.__name__
        if class_name.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif class_name.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)
        elif class_name not in [
            "AdaptiveAvgPool2d",
            "ModuleList",
            "ReLU",
            "Sequential",
            "UPerNet",
        ]:
            print("{} not initialized".format(class_name))

    def build_encoder(self, arch="resnet18", fc_dim=512, weights=""):
        pretrained = True if len(weights) == 0 else False
        print("Using pretrained weights: {}".format(pretrained))
        if arch == "resnet18":
            orig_resnet = get_resnet50(pretrained=pretrained)
            net_encoder = ResNet(orig_resnet)
        else:
            raise NotImplementedError

        if len(weights) > 0:
            print("Loading weights for net_encoder")
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage),
                strict=False,
            )
        return net_encoder

    def build_decoder(self, arch="upernet", fc_dim=2048, num_class=150, weights=""):
        if arch == "upernet":
            net_decoder = UPerNet(num_class=num_class, fc_dim=fc_dim, fpn_dim=256)
        else:
            raise NotImplementedError

        net_decoder.apply(self.init_weights)
        if len(weights) > 0:
            print("Loading weights for net_decoder")
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage),
                strict=False,
            )
        return net_decoder
