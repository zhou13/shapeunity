import math
from itertools import combinations
from collections import defaultdict

import cvxpy as cvx
import numpy as np
import numpy.linalg as LA
import scipy.optimize
import matplotlib.pyplot as plt

PI2 = math.pi * 2
MAX_ANGLE = PI2 / 8 / 10
cmap = plt.get_cmap("brg")


def estimate_intrinsic_from_depth(vertices, line):
    edges = defaultdict(list)
    for v0, v1 in line:
        edges[v0].append(v1)
        edges[v1].append(v0)
    for v in edges.copy():
        if len(edges[v]) == 1:
            del edges[v]

    def objective(x):
        inv2f = np.array([x[0], x[0], 1])
        o = 0
        for v0 in edges:
            for v1, v2 in combinations(edges[v0], 2):
                dv1 = vertices[v1] - vertices[v0]
                dv2 = vertices[v2] - vertices[v0]
                o += abs((inv2f * dv1) @ dv2)
        return o

    inv2f = scipy.optimize.minimize(objective, np.array([1]), options={"disp": True}).x
    assert (inv2f > 0).all()
    f = (1 / inv2f) ** 0.5
    return np.array([[f[0], 0, 0], [0, f[0], 0], [0, 0, 1]])


def estimate_intrinsic_from_vp(vp1, vp2, vp3):
    def objective(x):
        inv2f = np.array([x[0], x[0], 1])
        o = 0
        o += abs((inv2f * vp1) @ vp2)
        o += abs((inv2f * vp2) @ vp3)
        o += abs((inv2f * vp3) @ vp1)
        return o

    inv2f = scipy.optimize.minimize(objective, np.array([1]), method="COBYLA").x
    if inv2f[0] < 0:
        return None, objective(inv2f)
    f = (1 / inv2f) ** 0.5
    return np.array([[f[0], 0, 0], [0, f[0], 0], [0, 0, 1]]), objective(inv2f)


def to_world(junctions, juncdepth, lines, invK):
    vertices = np.c_[junctions, np.ones(len(junctions))]
    vertices *= juncdepth[:, None]
    vertices = vertices @ invK
    return (
        vertices,
        np.array(
            [
                [1 / invK[0, 0], 0, 0, 0],
                [0, 1 / invK[1, 1], 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
            ],
            dtype=np.float32,
        ),
    )


def vanish_point_clustering(junctions, lines):
    junctions_ = np.concatenate([junctions, np.ones([junctions.shape[0], 1])], axis=1)
    normals, weights = [], []
    for i1, i2 in lines:
        n = np.cross(junctions_[i1], junctions_[i2])
        n = n / LA.norm(n)
        normals.append(n)
        weights.append(LA.norm(junctions[i1] - junctions[i2]))
    normals, weights = np.array(normals), np.array(weights)
    weights /= np.amax(weights)

    clusters = []
    for i, (n1, (i1, i2)) in enumerate(zip(normals, lines)):
        for j, (n2, (j1, j2)) in enumerate(zip(normals[:i], lines[:i])):
            w = np.cross(n1, n2)
            if LA.norm(w) < 1e-4:
                continue
            w /= LA.norm(w)
            theta = np.abs(np.arcsin((w[None, :] * normals).sum(axis=1)))
            c = np.where(theta < PI2 / 8 / 15)[0]
            intersected = False
            if not intersected and len(c) >= 7:
                clusters.append((w, c))
                # plt.scatter(w[0] / w[2], w[1] / w[2])
                # for l in c:
                #     v1, v2 = junctions[lines[l][0]], junctions[lines[l][1]]
                #     plt.plot([v1[0], v2[0]], [v1[1], v2[1]])
                # plt.show()

    vp = []
    while True:
        w0, c = max(clusters, key=lambda x: len(x[1]) + weights[x[1]].sum())
        sc = set(c)
        w = np.zeros(3)
        weight = 0
        for l1 in c:
            for l2 in c:
                if l1 == l2:
                    continue
                w_ = np.cross(normals[l1], normals[l2])
                if w_ @ w > 0:
                    w += w_
                else:
                    w -= w_
                weight += LA.norm(w_)
        w /= weight

        vp.append((w, c))
        clusters = [
            (ww, list(set(cc) - sc)) for ww, cc in clusters if len(set(cc) - sc) >= 7
        ]
        if len(clusters) == 0:
            break

    plt.figure()
    for i, (w, c) in enumerate(vp):
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.scatter(w[0] / w[2], w[1] / w[2], c=cmap(i / len(vp)))
        for l in c:
            v1, v2 = junctions[lines[l][0]], junctions[lines[l][1]]
            plt.plot([v1[0], v2[0]], [v1[1], v2[1]], c=cmap(i / len(vp)))
    plt.show()
    return vp


def vanish_point_clustering2(junctions, lines):
    """Improved version of orthogonal vanish point clustering"""

    junctions_ = np.concatenate([junctions, np.ones([junctions.shape[0], 1])], axis=1)
    normals, weights = [], []

    def nearby_lines(w):
        theta = np.abs(np.arcsin((w[None, :] * normals).sum(axis=1)))
        return np.where(theta < MAX_ANGLE)[0]

    for i1, i2 in lines:
        n = np.cross(junctions_[i1], junctions_[i2])
        n = n / LA.norm(n)
        normals.append(n)
        weights.append(LA.norm(junctions[i1] - junctions[i2]))
    normals, weights = np.array(normals), np.array(weights)
    weights /= np.amax(weights)

    candidates = set([i for i in range(len(lines)) if weights[i] > 0.1])
    clusters = []
    for i, j in combinations(candidates, 2):
        w = np.cross(normals[i], normals[j])
        if LA.norm(w) < 1e-4:
            continue
        w /= LA.norm(w)
        line_candidates = set(nearby_lines(w)) & candidates
        if len(line_candidates) > 4:
            w = np.zeros(3)
            for p, q in combinations(line_candidates, 2):
                wp = np.cross(normals[p], normals[q])
                w += wp if wp @ w > 0 else -wp
            w /= LA.norm(w)
            if all(math.acos(abs(w @ wp * 0.99)) > 2 * MAX_ANGLE for wp, _ in clusters):
                clusters.append((w, line_candidates))

    tbd = set()
    for i, j in combinations(range(len(clusters)), 2):
        if i in tbd or j in tbd:
            continue
        c1 = clusters[i][1]
        c2 = clusters[j][1]
        if c1 >= c2:
            tbd.add(j)
        elif c1 <= c2:
            tbd.add(i)

    adj = defaultdict(list)
    for lineid in candidates:
        v1, v2 = lines[lineid]
        adj[v1].append(lineid)
        adj[v2].append(lineid)
    for i in range(len(clusters)):
        if i in tbd:
            continue
        for ls in adj.values():
            count = 0
            for l in ls:
                if l in clusters[i][1]:
                    count += 1
            if count > 1:
                tbd.add(i)
                break
    clusters = [clusters[i] for i in range(len(clusters)) if i not in tbd]

    print("Len of clusters", len(clusters), clusters)
    for i, (w, c) in enumerate(clusters):
        plt.figure()
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.scatter(w[0] / w[2], w[1] / w[2], c=cmap(i / len(clusters)))
        for l in c:
            v1, v2 = junctions[lines[l][0]], junctions[lines[l][1]]
            plt.plot([v1[0], v2[0]], [v1[1], v2[1]])

    W = []
    if len(clusters) < 3:
        W = [c for c, _ in clusters]
    else:
        best_cost = 1e8
        best_coverage = 0
        for (w1, c1), (w2, c2), (w3, c3) in combinations(clusters, 3):
            if max(len(c1 & c2), len(c2 & c3), len(c3 & c1)) > 2:
                continue
            coverage = len(c1 | c2 | c3)
            if coverage < best_coverage - 1:
                continue
            K, cost = estimate_intrinsic_from_vp(w1, w2, w3)
            if K is None:
                continue
            if 0.5 <= K[0, 0] < 10 and cost < best_cost:
                W = [w1, w2, w3]
                best_cost = cost
                best_coverage = coverage
                print(K, cost)

    vp = [(w, set()) for w in W]
    for i in sorted(list(candidates)):
        best = MAX_ANGLE * 5
        bestc = None
        for w, c in vp:
            degree = abs(math.asin(w @ normals[i]))
            if degree < best:
                best = degree
                bestc = c
        if bestc is not None:
            bestc.add(i)

    plt.figure()
    for i, (w, c) in enumerate(vp):
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.scatter(w[0] / w[2], w[1] / w[2], c=cmap(i / len(vp)))
        for l in c:
            v1, v2 = junctions[lines[l][0]], junctions[lines[l][1]]
            plt.plot([v1[0], v2[0]], [v1[1], v2[1]], c=cmap(i / len(vp)))
    plt.show()

    return vp


def lifting_from_vp(vp, invK, junctions, juncdepth, junctypes, lines, lambda_=2e0):
    vertices = np.c_[junctions, np.ones(len(junctions))] @ invK.T

    def objective(depth):
        o = 0
        for w, c in vp:
            w = invK @ w
            w /= LA.norm(w)
            for l in c:
                i, j = lines[l]
                uv = depth[i] * vertices[i] - depth[j] * vertices[j]
                o += LA.norm(uv - (uv @ w) * w)
        return o

    def cross(a, b):
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    def make_cvx_problem():
        depth = cvx.Variable(len(juncdepth))
        scale = cvx.Variable()
        objective = 0
        constraints = [depth >= 0.1, scale >= 0]
        for w, c in vp:
            w = invK @ w
            w /= LA.norm(w)
            for l in c:
                i, j = lines[l]
                uv = depth[i] * vertices[i] - depth[j] * vertices[j]
                objective += cvx.norm(cvx.hstack(cross(uv, w)))
        objective += lambda_ * cvx.sum_squares(depth - scale * juncdepth)
        problem = cvx.Problem(cvx.Minimize(objective), constraints)
        return problem, depth, scale

    problem, depth, scale = make_cvx_problem()
    problem.solve(solver="SCS")

    print(scale.value, 1 / invK[0, 0])
    return depth.value / scale.value
