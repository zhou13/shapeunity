#!/usr/bin/env python3
"""Evaluate result
Usage:
    eval_2d3d_metric.py [options] <npzdir> [<indices>...]
    eval_2d3d_metric.py (-h | --help )

Options:
   -h --help                 Show this screen.
   --show                    Show the result on screen
   -j --jobs <jobs>          Number of threads for vectorization [default: 1]
   --vpdir <vpdir>           Directory to the vanishing points prediction
                             [Default: logs/pretrained-vanishing-points/npz/000096000]
"""

import os
import sys
import json
import math
import random
import os.path as osp
from collections import deque

import cv2
import yaml
import numpy as np
import matplotlib as mpl
import skimage.io
import numpy.linalg as LA
import skimage.draw
import matplotlib.cm as cm
import skimage.filters
import matplotlib.pyplot as plt
import skimage.morphology
from docopt import docopt
# from tqdm import tqdm

from wireframe.utils import parmap
from wireframe.metric import nms_junction
from wireframe.viewer import show_wireframe
from vectorize_u3d import extract_wireframe
from wireframe.optimize import (
    to_world,
    lifting_from_vp,
    vanish_point_refine,
    vanish_point_clustering,
    vanish_point_clustering2,
    estimate_intrinsic_from_vp,
)


def ap(tp, fp):
    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[i + 1] - recall[i]) * precision[i + 1])


def msTPFP_hit(pt_line, gt_line, pt_3dline, gt_3dline, threshold, theta_t=90):
    diff = ((pt_line[:, None, :, None] - gt_line[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )
    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    # print("ptline", pt_line)
    # print("gtline", gt_line)
    if theta_t < 90:
        ptv = pt_3dline[:, 0] - pt_3dline[:, 1]  # (n, 3)
        gtv = gt_3dline[:, 0] - gt_3dline[:, 1]  # (m, 3)
        ptv_n = np.linalg.norm(ptv, axis=1)
        gtv_n = np.linalg.norm(gtv, axis=1)
        # print(ptv)
        # print(ptv_n)
        norm_matrix = ptv_n[:, None] * gtv_n[None]  # (n, m)
        norm_maxtrx_0 = norm_matrix == 0
        norm_maxtrx_non0 = ~norm_maxtrx_0
        inner_ptgt = np.sum(ptv[:, None] * gtv[None], axis=-1)
        cos = np.clip((inner_ptgt * norm_maxtrx_non0) / (norm_matrix + norm_maxtrx_0), 0, 1)
        theta = np.arccos(cos) * 180 / math.pi
    else:
        theta = np.zeros_like(diff)

    hit2 = np.zeros(len(gt_line), np.bool)
    tp2 = np.zeros(len(pt_line), np.float)
    fp2 = np.zeros(len(pt_line), np.float)

    hit3 = np.zeros(len(gt_line), np.bool)
    tp3 = np.zeros(len(pt_line), np.float)
    fp3 = np.zeros(len(pt_line), np.float)
    for i in range(len(pt_line)):
        if dist[i] < threshold and theta[i, choice[i]] < theta_t and not hit3[choice[i]]:
            hit3[choice[i]] = True
            tp3[i] = 1
        else:
            fp3[i] = 1

        if dist[i] < threshold and not hit2[choice[i]]:
            hit2[choice[i]] = True
            tp2[i] = 1
        else:
            fp2[i] = 1
    return tp2, fp2, hit2, tp3, fp3, hit3


def extract(index, datadir, npzdir, vpdir):
    batch = index // 100 + 1
    image = skimage.io.imread(f"{datadir}/{batch:03}/{index % 100:04}.png")
    result = np.load(f"{npzdir}/{index:06}.npz")
    print(f"Extracting {npzdir}/{index:06}.npz")

    with open(f"{datadir}/{batch:03}/{index % 100:04}_label.json") as f:
        js = json.load(f)
    gjunctions = js["junctions"]
    gjuncdepth = js["juncdepth"]
    gjunctypes = js["junctypes"]
    glines = js["lines"]

    gj512 = np.array(
        [
            256 * (1 + np.array(gjunctions)[:, 0]),
            256 * (1 - np.array(gjunctions)[:, 1]),
        ]
    ).T

    gt_lines = np.zeros((len(glines), 2, 2))
    for k, (i, j) in enumerate(glines):
        p1, p2 = gj512[i], gj512[j]
        gt_lines[k, 0] = p1
        gt_lines[k, 1] = p2

    os.makedirs(f"{npzdir}/wireframe", exist_ok=True)

    os.makedirs(f"{npzdir}/wireframe", exist_ok=True)
    junctions, junctypes, juncdepth, lines, edges = extract_wireframe(
        f"{npzdir}/wireframe/{index:06}",
        image,
        result,
        plot=False,
        imshow=False,
    )
    # gdepth = []
    # for jun in junctions:
    #     best_distance = 1e10
    #     best_i = 0
    #     for i, gjun in enumerate(gjunctions):
    #         if LA.norm(jun - gjun) < best_distance:
    #             best_distance = LA.norm(jun - gjun)
    #             best_i = i
    #     gdepth.append(gjuncdepth[best_i])
    # gdepth = np.array(gdepth)

    # FIXME: retrain the neural network to use the same npz
    vpfn = vpdir + f"/{index:06}.npz"
    vps = np.load(vpfn)["vpts"]

    # vps = vanish_point_clustering2(np.array(junctions), lines)
    vps = vanish_point_refine(np.array(junctions), np.array(lines), vps)

    # K = estimate_intrinsic_from_vp(vps[0][0], vps[1][0], vps[2][0])[0]
    # print("K:", K)
    # invK = LA.inv(K)

    K = np.array([[2.1875, 0, 0], [0, 2.1875, 0], [0, 0, 1]])
    invK = np.array([[0.45, 0, 0], [0, 0.45, 0], [0, 0, 1]])

    vertices_gt, _ = to_world(np.array(gjunctions), np.array(gjuncdepth), glines, K)
    gt_3dlines = vertices_gt[np.array(glines).reshape(-1)].reshape(-1, 2, 3)

    if len(junctions) == 0:
        return gt_lines, np.zeros((2, 2, 2)), np.zeros((2,)), gt_3dlines, np.zeros((2, 2, 3))

    depth = lifting_from_vp(vps, invK, junctions, -juncdepth, junctypes, lines)
    vertices_pt, _ = to_world(junctions, depth, lines, K)
    pt_3dlines = vertices_pt[np.array(lines).reshape(-1)].reshape(-1, 2, 3)

    pt_lines = np.zeros((len(lines), 2, 2))
    scores = []
    for k, e in enumerate(edges):
        p1, p2, score = e[0], e[1], e[2]
        pt_lines[k, 0] = p1[:2]
        pt_lines[k, 1] = p2[:2]
        scores.append(score)

    scores = np.array(scores)

    return gt_lines, pt_lines, scores, gt_3dlines, pt_3dlines


def main():
    args = docopt(__doc__)
    npzdir = args["<npzdir>"]
    indices = args["<indices>"] or [i for i in range(0, 300)]
    threshold_2d = 10
    threshold_3d = 10

    print(indices)

    with open(f"{npzdir}/../../config.yaml", "r") as f:
        c = yaml.load(f)
    datadir = c["io"]["datadir"]

    if int(args["--jobs"]) == 1:
        n_gt = 0
        n_pt = 0
        tps2, fps2, tps3, fps3, scores = [], [], [], [], []
        for index in map(int, indices):
            gt_lines, pt_lines, score, gt_3dlines, pt_3dlines = extract(index, datadir, npzdir, args["--vpdir"])
            n_gt += len(gt_lines)
            n_pt += len(pt_lines)
            tp2, fp2, _, tp3, fp3, _ = msTPFP_hit(pt_lines/4, gt_lines/4, pt_3dlines, gt_3dlines, threshold=threshold_2d, theta_t=threshold_3d)

            tps2.append(tp2)
            fps2.append(fp2)

            tps3.append(tp3)
            fps3.append(fp3)

            scores.append(score)

        tps2 = np.concatenate(tps2)
        fps2 = np.concatenate(fps2)

        tps3 = np.concatenate(tps3)
        fps3 = np.concatenate(fps3)

        scores = np.concatenate(scores)
        index = np.argsort(-scores)
        tp_2d = np.cumsum(tps2[index]) / n_gt
        fp_2d = np.cumsum(fps2[index]) / n_gt

        tp_3d = np.cumsum(tps3[index]) / n_gt
        fp_3d = np.cumsum(fps3[index]) / n_gt

        sap_2d = ap(tp_2d, fp_2d)
        sap_3d = ap(tp_3d, fp_3d)

        print(f"2D metric sAP-{threshold_2d}: {sap_2d}. \n 3D metric sAP-{threshold_2d}-{threshold_3d}: {sap_3d}. ")
        with open(f"metric2d3d.csv", "a") as fout:
            print(f"2D metric sAP-{threshold_2d}: {sap_2d}. \n 3D metric sAP-{threshold_2d}-{threshold_3d}: {sap_3d}. ", file=fout)
    else:
        raise ValueError("not implementation")


if __name__ == "__main__":
    np.seterr(all="raise")
    plt.rcParams["figure.figsize"] = (8, 8)
    main()