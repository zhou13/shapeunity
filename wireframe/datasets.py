import os
import glob
import json
import math
import random
import os.path as osp
from timeit import default_timer as timer
from collections import deque

import cv2
import numpy as np
import torch
import numpy.linalg as LA
import matplotlib.pylab as plt
from skimage import io, transform
from torch.utils.data import Dataset

import wireframe.utils as utils


class WireframeDataset(Dataset):
    def __init__(
        self,
        rootdir,
        split,
        image_mean=(109.730, 103.832, 98.681),
        image_std=(22.275, 22.124, 23.229),
    ):
        self.rootdir = rootdir
        filelist = glob.glob(f"{rootdir}/[0-9]*/*_label.npz")
        filelist.sort()
        if split == "train":
            filelist = filelist[600:]
            print("ntrain:", len(filelist))
        elif split == "valid":
            filelist = [f for f in filelist[:600] if "flip" not in f]
        else:
            assert False
        self.split = split
        self.filelist = filelist
        self.image_mean = image_mean
        self.image_std = image_std

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        iname = self.filelist[idx][:-10].replace("_flip", "") + ".png"
        image = io.imread(iname).astype(float)[:, :, :3]
        image = (image - self.image_mean) / self.image_std
        if "flip" in self.filelist[idx]:
            image = image[:, ::-1, :]
        image = np.rollaxis(image, 2).copy()
        npz = np.load(self.filelist[idx])
        return (
            torch.from_numpy(image).float(),
            {name: torch.from_numpy(arr).float() for name, arr in npz.items()},
        )

    def ground_truth(self, idx):
        assert self.split == "valid"
        with open(self.filelist[idx].replace("npz", "json")) as fin:
            js = json.load(fin)

        weights = {}
        junc = js["junctions"]
        theta_weights = [[] for _ in range(len(junc))]
        jun_theta = [[] for _ in range(len(junc))]
        for j0, j1 in js["lines"]:
            v0, v1 = junc[j0], junc[j1]
            dist = LA.norm(
                np.array(js["junctions"][j0]) - np.array(js["junctions"][j1])
            )
            weights[j0] = weights[j0] + dist if j0 in weights else dist
            weights[j1] = weights[j1] + dist if j1 in weights else dist
            theta = math.atan2(-v1[1] + v0[1], v1[0] - v0[0])
            jun_theta[j0].append(theta)
            jun_theta[j1].append(theta + math.pi)
            theta_weights[j0].append(dist)
            theta_weights[j1].append(dist)

        jun = [[], []]
        jun_w = [[], []]
        jun_dir = [[], []]
        jun_dir_weight = [[], []]
        jun_d = [[], []]
        for idx, (j, typ) in enumerate(zip(js["junctions"], js["junctypes"])):
            jun[typ].append([(1 - j[1]) * 64 - 0.5, (1 + j[0]) * 64 - 0.5])
            jun_w[typ].append(weights[idx])
            jun_dir[typ].append(jun_theta[idx][-1] if typ == 1
                                else jun_theta[idx])
            jun_dir_weight[typ].append(theta_weights[idx][-1] if typ == 1
                                       else theta_weights[idx])
            jun_d[typ].append(js["juncdepth"][idx])

        return (
            np.array(jun[0]),
            np.array(jun[1]),
            np.array(jun_w[0]),
            np.array(jun_w[1]),
            np.array(jun_d[0]),
            np.array(jun_d[1]),
        )
