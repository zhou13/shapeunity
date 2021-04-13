#!/usr/bin/env python3
"""Vectorize result
Usage:
    vectorize.py [options] <npzdir> [<indices>...]
    vectorize.py (-h | --help )

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

from wireframe.utils import parmap
from wireframe.metric import nms_junction
from wireframe.viewer import show_wireframe
from wireframe.optimize import (
    to_world,
    lifting_from_vp,
    vanish_point_refine,
    vanish_point_clustering,
    vanish_point_clustering2,
    estimate_intrinsic_from_vp,
)

PI2 = math.pi * 2
NMS_ANGLE = PI2 / 24
JUNC = 0.2
JUND = 0.3

MAX_T_DISTANCE = 5
T_SCALE = 1.1

# Thresholding
MEDIAN = 0.1
SCORE = 0.65
N_ITER = 3

# Gaussian blur
SIGMA = 0.5
SCALE = 2.0


# setup matplotlib
cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.6, vmax=1.1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def line_color(x):
    return sm.to_rgba(x)


def filter_heatmap(h, h_threshold, offset=None):
    if offset is None:
        offset = np.zeros([2] + list(h.shape))
    result = []
    for y in range(h.shape[0]):
        for x in range(h.shape[1]):
            if h[y, x] > h_threshold:
                result.append(
                    [(x + offset[0, y, x]) * 4, (y + offset[1, y, x]) * 4, h[y, x]]
                )

    return result


def project(c, a, b):
    px = b[0] - a[0]
    py = b[1] - a[1]
    dd = px * px + py * py
    u = max(min(((c[0] - a[0]) * px + (c[1] - a[1]) * py) / float(dd), 1), 0)
    return (a[0] + u * px, a[1] + u * py)


def is_intersected(a0, a1, b0, b1):
    def ccw(c, a, b):
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])

    def sgn(x):
        if abs(x) < 1e-6:
            return 0
        if x > 0:
            return 1
        return -1

    c0 = sgn(ccw(a0, a1, b0))
    c1 = sgn(ccw(a0, a1, b1))
    d0 = sgn(ccw(b0, b1, a0))
    d1 = sgn(ccw(b0, b1, a1))
    return c0 * c1 < 0 and d0 * d1 < 0


def angle(c, a, b):
    a = (a[0] - c[0], a[1] - c[1])
    b = (b[0] - c[0], b[1] - c[1])
    dot = (
        (a[0] * b[0] + a[1] * b[1])
        / math.sqrt(a[0] ** 2 + a[1] ** 2 + 1e-9)
        / math.sqrt(b[0] ** 2 + b[1] ** 2 + 1e-9)
    )
    return math.acos(max(min(dot, 1), -1))


def point2line(c, a, b):
    px = b[0] - a[0]
    py = b[1] - a[1]
    dd = px * px + py * py
    u = ((c[0] - a[0]) * px + (c[1] - a[1]) * py) / float(dd)
    if u <= 0 or u >= 1:
        return 100
    dx = a[0] + u * px - c[0]
    dy = a[1] + u * py - c[1]
    return dx * dx + dy * dy


def parse_result(result):
    junc = nms_junction(result["jmap"][0])
    jund = nms_junction(result["jmap"][1])
    line = result["lmap"]
    jdep = result["jdep"]

    junc = filter_heatmap(junc, JUNC)
    jund = filter_heatmap(jund, JUND)
    jun = junc + jund

    return jun, list(range(len(junc))), list(range(len(junc), len(jun))), line, jdep


def edge_pruning(juncs, edges):
    def polar_angle(p0, p1):
        return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

    def polar_diff(p1, p2):
        d = math.fmod(p1 - p2, PI2)
        if d < 0:
            d += PI2
        return min(abs(d), abs(PI2 - d))

    esets = set()
    links = [{} for _ in range(len(juncs))]

    def delete_edge(i):
        j1, j2 = edges[i][3], edges[i][4]
        del links[j1][i]
        del links[j2][i]
        esets.remove(i)

    for it in range(N_ITER):
        for i, (p1, p2, score, j1, j2) in enumerate(edges):
            if i in links[j1]:
                continue
            angle1 = polar_angle(p1, p2)
            angle2 = math.fmod(angle1 + math.pi, PI2)

            # check nearby edges
            error = False
            # if it == N_ITER - 1:
            #     score = -1
            for j, angle in links[j1].copy().items():
                if polar_diff(angle, angle1) < NMS_ANGLE and edges[j][2] > score:
                    error = True
                    break
            if error:
                continue
            for j, angle in links[j2].copy().items():
                if polar_diff(angle, angle2) < NMS_ANGLE and edges[j][2] > score:
                    error = True
                    break
            if error:
                continue

            # prunning other edges
            for j, angle in links[j1].copy().items():
                if polar_diff(angle, angle1) < NMS_ANGLE and edges[j][2] < score:
                    delete_edge(j)
            for j, angle in links[j2].copy().items():
                if polar_diff(angle, angle2) < NMS_ANGLE and edges[j][2] < score:
                    delete_edge(j)

            # add this edge
            esets.add(i)
            links[j1][i] = angle1
            links[j2][i] = angle2

        # remove intersected edges
        for i in esets.copy():
            if i not in esets:
                continue
            for j in esets.copy():
                if j not in esets:
                    continue
                if edges[i][2] < edges[j][2]:
                    continue
                if is_intersected(*edges[i][:2], *edges[j][:2]):
                    delete_edge(j)

    return [edges[i] for i in sorted(esets)]


def line_score(p1, p2, line_map, shrink=True):
    if p1[0] == p2[0] and p1[1] == p2[1]:
        return -1, -1
    r0, c0, r1, c1 = map(int, [p1[1] // 4, p1[0] // 4, p2[1] // 4, p2[0] // 4])
    rr, cc, I = skimage.draw.line_aa(r0, c0, r1, c1)
    if shrink:
        if len(rr) <= 2:
            return -1, -1
        rr, cc, I = rr[1:-1], cc[1:-1], I[1:-1]
    Ip = line_map[rr, cc]
    Ip = Ip / np.maximum(I, Ip)
    score = (I * Ip).sum() / I.sum()
    Ip_sorted = np.sort(Ip)
    median = Ip_sorted[max(min(2, len(Ip) - 1), len(Ip) // 7)]
    return score, median


def extract_wireframe(prefix, image, result, plot=True, imshow=True):
    jun, ijunc, ijund, line_map, jdep = parse_result(result)

    line_map[line_map > 1] = 1
    line_map = skimage.filters.gaussian(line_map, SIGMA) * SCALE
    line_map[line_map > 1] = 1
    if plot:
        # plt.figure(), plt.imshow(jdep[0])
        # plt.figure(), plt.imshow(jdep[1])

        # plt.figure(), plt.title("Edge map"), plt.tight_layout()
        # plt.imshow(line_map), plt.colorbar(fraction=0.046)
        plt.figure(), plt.axis("off"), plt.tight_layout(), plt.axes([0, 0, 1, 1])
        plt.xlim([-0.5, 127.5]), plt.ylim([127.5, -0.5])
        plt.imshow(line_map, cmap="Purples")
        # for i in ijunc:
        #     plt.scatter(jun[i][0] / 4, jun[i][1] / 4, color="red", zorder=100)
        # for i in ijund:
        #     plt.scatter(jun[i][0] / 4, jun[i][1] / 4, color="blue", zorder=100)
        plt.savefig(f"{prefix}_map.svg", bbox_inches=0)
        plt.close()

        # plt.figure(), plt.title("Initial Wireframe"), plt.tight_layout()
        # plt.imshow(image), plt.colorbar(sm, fraction=0.046)
        # for i in ijunc:
        #     plt.scatter(jun[i][0], jun[i][1], color="red", zorder=100)
    edges = []
    for i_, i in enumerate(ijunc):
        for j in ijunc[:i_]:
            p1, p2 = jun[i], jun[j]
            score, median = line_score(p1, p2, line_map)
            if median > MEDIAN and score > SCORE:
                edges.append((p1, p2, score, i, j))
                # plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c=line_color(score))

    edges.sort(key=lambda e: e[2])
    edges = edge_pruning(jun, edges)

    # plt.figure(), plt.title("Prunned Wireframe"), plt.tight_layout()
    # plt.imshow(image), plt.colorbar(sm, fraction=0.046)
    # for i in ijunc:
    #     plt.scatter(jun[i][0], jun[i][1], color="red", zorder=100)
    # for p1, p2, score, *_ in edges:
    #     plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c=line_color(score))
    selected_juns = set(i for i in ijunc)
    for i in ijund:
        pi = jun[i][:2]
        mind = 1e10
        for e in edges:
            dist = point2line(pi, e[0], e[1])
            if dist < mind:
                mind = dist
                mine = e
        if mind < MAX_T_DISTANCE:
            pip = project(pi, mine[0], mine[1])  # reproject for nicer figure
            jun[i][0], jun[i][1] = pip[0], pip[1]
            best_score = -1e100
            for j, pj in enumerate(jun):
                if i == j or (j > len(ijunc) and j not in selected_juns):
                    continue
                if min(angle(pi, pj, mine[0]), angle(pi, pj, mine[1])) < 0.2:
                    continue
                if LA.norm(np.array(pi[:2]) - pj[:2]) < 12:
                    continue
                score, median = line_score(pi, pj, line_map, shrink=False)
                if median > MEDIAN and score > best_score:
                    best_score = score
                    bestj = j
                    bestp = pj
            if best_score > SCORE:
                edges.append((pip, bestp, best_score * T_SCALE, i, bestj))
            selected_juns.add(i)

    edges.sort(key=lambda e: e[2])
    edges = edge_pruning(jun, edges)
    selected_juns = set()
    for *_, i, j in edges:
        selected_juns.add(i)
        selected_juns.add(j)

    if plot:
        # plt.figure(), plt.title("Final Wireframe"), plt.tight_layout()
        # plt.imshow(image), plt.colorbar(sm, fraction=0.046)
        plt.figure(), plt.axis("off"), plt.tight_layout(), plt.axes([0, 0, 1, 1])
        plt.xlim([-0.5, 511.5]), plt.ylim([511.5, -0.5])
        plt.imshow(image)
        for i in ijunc:
            alpha = 1 if i in selected_juns else 0.5
            plt.scatter(
                jun[i][0], jun[i][1], color="red", alpha=alpha, zorder=100, s=40
            )
        for i in ijund:
            alpha = 1 if i in selected_juns else 0.5
            plt.scatter(
                jun[i][0], jun[i][1], color="blue", alpha=alpha, zorder=100, s=40
            )
        for e in edges:
            p1, p2, score = e[0], e[1], e[2]
            # plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c=line_color(score))
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c="darkorange", linewidth=2)
        plt.savefig(f"{prefix}.svg")
        if imshow:
            plt.show()
        plt.close()

    junctions = []
    junctypes = []
    juncdepth = []

    new_index = [0] * len(jun)
    index = 0
    for i in selected_juns:
        new_index[i] = index
        junctions.append([jun[i][0] / 256 - 1, 1 - jun[i][1] / 256])
        junctypes.append(int(i in ijund))
        juncdepth.append(jdep[int(i in ijund)][int(jun[i][1] / 4), int(jun[i][0] / 4)])
        index += 1
    lines = [[new_index[i], new_index[j]] for *_, i, j in edges]

    return np.array(junctions), np.array(junctypes), np.array(juncdepth), lines, edges


def main():
    args = docopt(__doc__)
    npzdir = args["<npzdir>"]
    indices = args["<indices>"] or [101]
    with open(f"{npzdir}/../../config.yaml", "r") as f:
        c = yaml.load(f, Loader=yaml.FullLoader)
    datadir = c["io"]["datadir"]

    def extract(index):
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
        plt.figure(), plt.axis("off"), plt.tight_layout(), plt.axes([0, 0, 1, 1])
        plt.xlim([-0.5, 511.5]), plt.ylim([511.5, -0.5])
        plt.imshow(image)
        os.makedirs(f"{npzdir}/wireframe", exist_ok=True)
        for junc, typ in zip(gj512, gjunctypes):
            if typ == 0:
                color = "red"
            else:
                color = "blue"
            plt.scatter(junc[0], junc[1], color=color, zorder=100, s=40)
        for i, j in glines:
            p1, p2 = gj512[i], gj512[j]
            # plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c=line_color(score))
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c="darkorange", linewidth=2)
        plt.savefig(f"{npzdir}/wireframe/{index:06}_gt.svg")
        plt.close()

        os.makedirs(f"{npzdir}/wireframe", exist_ok=True)
        junctions, junctypes, juncdepth, lines, _ = extract_wireframe(
            f"{npzdir}/wireframe/{index:06}",
            image,
            result,
            plot=True,
            imshow=args["--show"],
        )
        gdepth = []
        for jun in junctions:
            best_distance = 1e10
            best_i = 0
            for i, gjun in enumerate(gjunctions):
                if LA.norm(jun - gjun) < best_distance:
                    best_distance = LA.norm(jun - gjun)
                    best_i = i
            gdepth.append(gjuncdepth[best_i])
        gdepth = np.array(gdepth)

        # FIXME: retrain the neural network to use the same npz
        vpfn = args["--vpdir"] + f"/{index:06}.npz"
        vps = np.load(vpfn)["vpts"]
        ## show vanish point
        # for vp in vps:
        #     vp_ = [256 * (1 + vp[0] / vp[2]) - 0.5, 256 * (1 - vp[1] / vp[2]) - 0.5]
        #     plt.scatter(vp_[0], vp_[1])
        # plt.show()

        vps = vanish_point_refine(np.array(junctions), np.array(lines), vps)
        # vps = vanish_point_clustering2(np.array(junctions), lines)
        K = estimate_intrinsic_from_vp(vps[0][0], vps[1][0], vps[2][0])[0]
        invK = LA.inv(K)
        print("K:", 1 / invK[0, 0])
        depth = lifting_from_vp(vps, invK, junctions, -juncdepth, junctypes, lines)
        vertices, projection_matrix = to_world(junctions, depth, lines, K)
        show_wireframe(
            f"{npzdir}/wireframe/{index:06}", vertices, lines, projection_matrix
        )
        return 0

    if args["--show"] or int(args["--jobs"]) == 1:
        for index in map(int, indices):
            extract(index)
    else:
        parmap(extract, map(int, indices), int(args["<jobs>"]))


if __name__ == "__main__":
    np.seterr(all="raise")
    plt.rcParams["figure.figsize"] = (8, 8)
    main()
