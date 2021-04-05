from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    """This loss will bias to false negative"""
    nlogp = -F.log_softmax(logits, dim=0)
    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


def masked_sine_loss(input, target, mask):
    w = mask.mean(2, True).mean(1, True)
    w[w == 0] = 1
    return (torch.sin(input - target) ** 2 * (mask / w)).mean(2).mean(1)


def masked_l2loss(input, target, mask):
    w = mask.mean(2, True).mean(1, True)
    w[w == 0] = 1
    return ((target - input) ** 2 * (mask / w)).mean(2).mean(1)


def sine_loss(input, target):
    return (torch.sin(input - target) ** 2).mean(2).mean(1)


def depth_loss(input, target):
    mask = (target > 0).float()
    invn = 1.0 / mask.sum((1, 2), True)
    d = mask * (input - torch.log(torch.clamp(target, min=1e-5)))
    loss = (d ** 2).sum((1, 2), True) * invn - (d.sum((1, 2), True) * invn) ** 2
    return loss.mean(2).mean(1)


class MultitaskLearner(nn.Module):
    NUM_CLASS = 45

    def __init__(self, backbone):
        super(MultitaskLearner, self).__init__()
        self.backbone = backbone

    def forward(self, input_dict):
        data = input_dict["data"]
        T = input_dict["target"]
        result = self.backbone(data)
        batch, channel, row, col = result[0].shape

        # switch to CNHW
        for task in ["jmap", "jwgt", "jdep"]:
            T[task] = T[task].permute(1, 0, 2, 3)
        for task in ["joff", "jdir"]:
            T[task] = T[task].permute(1, 2, 0, 3, 4)

        loss_weight = {
            "jmap": 1.0,
            "jdir": 1.0,
            "lmap": 5.0,
            "joff": 0.5,
            "ldir": 1.0,
            "dpth": 1.0,
            "jdep": 0.1,
        }
        losses = []
        for stack, input in enumerate(result):
            input = (
                input.transpose(0, 1)
                .reshape([MultitaskLearner.NUM_CLASS, batch, row, col])
                .contiguous()
            )

            jmap = input[0:4].reshape(2, 2, batch, row, col)
            jdir = input[4:36].reshape(2, 8, 2, batch, row, col)
            lmap = input[36]
            joff = input[37:41].reshape(2, 2, batch, row, col)
            ldir = input[41]
            dpth = input[42]
            jdep = input[43:45]
            if stack == 0 and input_dict["output_heatmap"]:
                heatmaps = {
                    "jmap": jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],
                    "jdir": jdir.permute(3, 0, 1, 2, 4, 5).softmax(3)[:, :, :, 1],
                    "lmap": lmap,
                    "joff": joff.permute(2, 0, 1, 3, 4),
                    "ldir": ldir,
                    "dpth": dpth,
                    "jdep": jdep.transpose(0, 1),
                }

            L = OrderedDict()
            L["jmap"] = sum(cross_entropy_loss(jmap[i], T["jmap"][i]) for i in range(2))
            L["jdir"] = sum(
                cross_entropy_loss(jdir[i, j], T["jdir"][i, j])
                for i in range(2)
                for j in range(8)
            )
            L["lmap"] = l2loss(lmap, T["lmap"])
            L["joff"] = sum(
                masked_l2loss(joff[i, j], T["joff"][i, j], T["jmap"][i])
                for i in range(2)
                for j in range(2)
            )
            L["ldir"] = sine_loss(ldir, T["ldir"])
            L["dpth"] = l2loss(dpth, T["dpth"])
            L["jdep"] = sum(
                masked_l2loss(jdep[i], T["jdep"][i], T["jmap"][i]) for i in range(2)
            )
            for loss_name in L:
                L[loss_name].mul_(loss_weight[loss_name])
            losses.append(L)

        if input_dict["output_heatmap"]:
            return {"heatmaps": heatmaps, "losses": losses}
        else:
            return {"losses": losses}
