import os
import glob
import json
import random
import os.path as osp
from timeit import default_timer as timer
from collections import deque

import cv2
import numpy as np
import torch
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
        filelist = glob.glob(f"{rootdir}/[0-9]*/*_label.json")
        filelist.sort()
        if split == "train":
            filelist = filelist[300:]
        elif split == "valid":
            filelist = filelist[:300]
        else:
            assert False
        self.split = split
        self.filelist = [s.replace("_label.json", "") for s in filelist]
        self.image_mean = image_mean
        self.image_std = image_std

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        image = io.imread(f"{self.filelist[idx]}.png").astype(float)[:, :, :3]
        image = (image - self.image_mean) / self.image_std
        image = np.rollaxis(image, 2)
        npz = np.load(f"{self.filelist[idx]}_label.npz")
        return (
            torch.from_numpy(image).float(),
            {
                "jmap": torch.from_numpy(npz["jmap"]).float(),
                "jwgt": torch.from_numpy(npz["jwgt"]).float(),
                "jdep": torch.from_numpy(npz["jdep"]).float(),
                "joff": torch.from_numpy(npz["joff"]).float(),
                "jdir": torch.from_numpy(npz["jdir"]).float(),
                "lmap": torch.from_numpy(npz["lmap"]).float(),
                "ldir": torch.from_numpy(npz["ldir"]).float(),
                "dpth": torch.from_numpy(npz["dpth"]).float(),
            },
        )

    def rawdata(self, idx):
        with open(f"{self.filelist[idx]}_label.json") as fin:
            js = json.load(fin)
        jun = [[], []]
        for j, typ in zip(js["junctions"], js["junctypes"]):
            jun[typ].append([(1 + j[0]) * 64 - 0.5, (1 - j[1]) * 64 - 0.5])
        return np.array(jun[0]), np.array(jun[1])
