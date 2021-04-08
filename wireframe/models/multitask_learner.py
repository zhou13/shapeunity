from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def l1loss(input, target):
    return torch.abs(target - input)


def cross_entropy_loss(logits, positive, normalizer="spatial"):
    """This loss will bias to false negative"""
    nlogp = -F.log_softmax(logits, dim=0)
    if normalizer == "spatial":
        return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)
    elif normalizer == "num_fg":  # num_fg shape: (n)
        num_fg = torch.clamp(positive.sum(dim=2).sum(dim=1), min=32)
        return (positive * nlogp[1] + (1 - positive) * nlogp[0]).sum(2).sum(1) / num_fg
    else:
        return NotImplementedError


def masked_sine_loss(input, target, mask, scale=1):
    w = mask.mean(2, True).mean(1, True)
    w[w == 0] = 1
    return (torch.abs(torch.sin(scale * (input - target))) * (mask / w)).mean(2).mean(1)


def masked_l2loss(input, target, mask):
    w = mask.mean(2, True).mean(1, True)
    w[w == 0] = 1
    return ((target - input) ** 2 * (mask / w)).mean(2).mean(1)


def masked_l1loss(input, target, mask):
    w = mask.mean(2, True).mean(1, True)
    w[w == 0] = 1
    return (torch.abs(target - input) * (mask / w)).mean(2).mean(1)


def sigmoid_l1_loss(logits, target, offset=0., mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)


def sine_loss(input, target):
    return (torch.sin(input - target) ** 2).mean(2).mean(1)


def depth_loss(input, target):
    mask = (target > 0).float()
    invn = 1.0 / mask.sum((1, 2), True)
    d = mask * (input - torch.log(torch.clamp(target, min=1e-5)))
    loss = (d ** 2).sum((1, 2), True) * invn - (d.sum((1, 2), True) * invn) ** 2
    return loss.mean(2).mean(1)


def chamfer_loss(pred, gt):
    z_dist = torch.min(
        ((pred[:, 2] - gt[:, 2]) ** 2).sum(1),
        ((pred[:, 2] + gt[:, 2]) ** 2).sum(1),
    )

    x_dist = torch.min(
        torch.min(
            ((pred[:, 0] - gt[:, 0]) ** 2).sum(1),
            ((pred[:, 0] + gt[:, 0]) ** 2).sum(1),
        ),
        torch.min(
            ((pred[:, 1] - gt[:, 0]) ** 2).sum(1),
            ((pred[:, 1] + gt[:, 0]) ** 2).sum(1),
        ),
    )

    y_dist = torch.min(
        torch.min(
            ((pred[:, 0] - gt[:, 1]) ** 2).sum(1),
            ((pred[:, 0] + gt[:, 1]) ** 2).sum(1),
        ),
        torch.min(
            ((pred[:, 1] - gt[:, 1]) ** 2).sum(1),
            ((pred[:, 1] + gt[:, 1]) ** 2).sum(1),
        ),
    )
    return y_dist + x_dist + z_dist


class MultitaskLearner(nn.Module):
    NUM_CLASS = 16

    def __init__(self, backbone):
        super(MultitaskLearner, self).__init__()
        self.backbone = backbone

    def forward(self, input_dict):
        data = input_dict["data"]
        T = input_dict["target"]
        result, vpts = self.backbone(data)
        batch, channel, row, col = result[0].shape

        # switch to CNHW
        for task in ["jwgt", "jmap", "jdep"]:
            T[task] = T[task].permute(1, 0, 2, 3)
        for task in ["joff"]:
            T[task] = T[task].permute(1, 2, 0, 3, 4)

        loss_weight = {
            "jmap": 2.0,
            # "cdir": 0.5,
            # "tdir": 0.1,
            "lmap": 3.0,
            "joff": 0.25,
            "ldir": 0.05,
            "dpth": 1.0,
            "jdep": 0.1,
            "xmap": 2.0,
            "ymap": 2.0,
            "zmap": 2.0,
            "vpts": 1.0
        }
        losses = []
        for stack, input in enumerate(result):
            input = (
                input.transpose(0, 1)
                .reshape([MultitaskLearner.NUM_CLASS, batch, row, col])
                .contiguous()
            )
            vpt = vpts[stack].reshape([batch, 3, 3])
            jmap = input[0:4].reshape(2, 2, batch, row, col)
            # cdir = input[4:36].reshape(16, 2, batch, row, col)
            lmap = input[4]
            joff = input[5:9].reshape(2, 2, batch, row, col)
            ldir = input[9]
            dpth = input[10]
            jdep = input[11:13]
            xmap = input[13]
            ymap = input[14]
            zmap = input[15]
            if stack == 0 and input_dict["output_heatmap"]:
                heatmaps = {
                    "jmap": jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],
                    # "cdir": cdir.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],
                    # "tdir": tdir,
                    "lmap": lmap.sigmoid(),
                    "joff": joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5,
                    "ldir": ldir,
                    "dpth": dpth,
                    "jdep": jdep.transpose(0, 1),
                    "xmap": xmap.sigmoid(),
                    "ymap": ymap.sigmoid(),
                    "zmap": zmap.sigmoid(),
                    "vpts": vpt,
                }

            L = OrderedDict()
            L["jmap"] = sum(
                cross_entropy_loss(jmap[i], T["jmap"][i])
                for i in range(2)
            )
            # L["cdir"] = sum(
            #     cross_entropy_loss(cdir[i], T["cdir"][i]) for i in range(16)
            # )
            # L["tdir"] = masked_sine_loss(tdir, T["tdir"], T["jmap"][1], scale=0.5)
            L["lmap"] = F.binary_cross_entropy_with_logits(lmap, T["lmap"], reduction='none').mean(2).mean(1)
            L["joff"] = sum(
                sigmoid_l1_loss(joff[i, j], T["joff"][i, j], -0.5, T["jmap"][i])
                for i in range(2)
                for j in range(2)
            )
            # L["ldir"] = masked_l2_loss_from_theta(ldir, T["ldir"], T["lmap"])
            L["ldir"] = masked_sine_loss(ldir, T["ldir"], T["lmap"])
            L["dpth"] = l1loss(dpth, T["dpth"])
            L["jdep"] = sum(
                masked_l1loss(jdep[i], T["jdep"][i], T["jmap"][i]) for i in range(2)
            )
            L["xmap"] = torch.min(F.binary_cross_entropy_with_logits(xmap, T["Lmap"][:, 0], reduction='none').mean(2).mean(1),
                                  F.binary_cross_entropy_with_logits(ymap, T["Lmap"][:, 0], reduction='none').mean(2).mean(1))
            L["ymap"] = torch.min(F.binary_cross_entropy_with_logits(xmap, T["Lmap"][:, 1], reduction='none').mean(2).mean(1),
                                  F.binary_cross_entropy_with_logits(ymap, T["Lmap"][:, 1], reduction='none').mean(2).mean(1))
            L["zmap"] = F.binary_cross_entropy_with_logits(zmap, T["Lmap"][:, 2], reduction='none').mean(2).mean(1)
            L["vpts"] = chamfer_loss(vpt, T["vpts"])
            for loss_name in L:
                L[loss_name].mul_(loss_weight[loss_name])
            losses.append(L)

        if input_dict["output_heatmap"]:
            return {"heatmaps": heatmaps, "losses": losses}
        else:
            return {"losses": losses}
