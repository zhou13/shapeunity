import math
from itertools import combinations
from collections import namedtuple, defaultdict

import cvxpy as cvx
import numpy as np
import numpy.linalg as LA
import scipy.optimize
import matplotlib.pyplot as plt

PI2 = math.pi * 2
MAX_ANGLE = PI2 / 8 / 10
cmap = plt.get_cmap("brg")


def vp_from_lines(x1, y1, x2, y2):
    x, y = cvx.Variable(), cvx.Variable()
    assert len(x1) == len(x2) == len(y1) == len(y2)
    objective = cvx.sum(
        cvx.abs(cvx.multiply(x2 - x1, y1 - y) - cvx.multiply(y2 - y1, x1 - x))
    )
    problem = cvx.Problem(cvx.Minimize(objective), [])
    problem.solve()
    return x.value, y.value


def estimate_intrinsic_from_depth(vertices, line):
    edges = defaultdict(list)
    for v0, v1 in line:
        edges[v0].append(v1)
        edges[v1].append(v0)
    for v in edges.copy():
        if len(edges[v]) == 1:
            del edges[v]

    def objective(x):
        inv2f = np.array([x[0], x[1], 1])
        o = 0
        for v0 in edges:
            for v1, v2 in combinations(edges[v0], 2):
                dv1 = vertices[v1] - vertices[v0]
                dv2 = vertices[v2] - vertices[v0]
                o += abs((inv2f * dv1) @ dv2)
        return o

    inv2f = scipy.optimize.minimize(
        objective, np.array([1, 1]), options={"disp": True}
    ).x
    assert (inv2f > 0).all()
    f = (1 / inv2f) ** 0.5
    return np.array([[f[0], 0, 0], [0, f[1], 0], [0, 0, 1]])


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
        return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]), objective(inv2f)
    f = (1 / inv2f) ** 0.5
    return np.array([[f[0], 0, 0], [0, f[0], 0], [0, 0, 1]]), objective(inv2f)


def estimate_3intrinsic_from_vp(vp1, vp2, vp3):
    def objective(x):
        S = np.array([[1, 0, -x[0]], [0, 1, -x[1]], [-x[0], -x[1], x[2]]])
        o = (vp1 @ S @ vp2) ** 2 + (vp2 @ S @ vp3) ** 2 + (vp3 @ S @ vp1) ** 2
        return o

    S = scipy.optimize.minimize(objective, np.array([1, 1, 1]), method="COBYLA").x
    ox, oy = S[0], S[1]
    assert S[2] - ox ** 2 - oy ** 2 > 0
    f = math.sqrt(S[2] - ox ** 2 - oy ** 2)
    return np.array([[f, 0, ox], [0, f, oy], [0, 0, 1]]), objective(S)


def to_world(junctions, juncdepth, lines, K):
    vertices = np.c_[junctions, np.ones(len(junctions))]
    vertices *= juncdepth[:, None]
    vertices = vertices @ LA.inv(K).T
    return (
        vertices,
        np.array(
            [
                [K[0, 0], 0, K[0, 2], 0],
                [0, K[1, 1], K[1, 2], 0],
                [0, 0, 0, -1],
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

    # plt.figure(), plt.tight_layout()
    # for i, (w, c) in enumerate(vp):
    #     plt.xlim([-5, 5])
    #     plt.ylim([-5, 5])
    #     plt.scatter(w[0] / w[2], w[1] / w[2], c=cmap(i / len(vp)))
    #     for l in c:
    #         v1, v2 = junctions[lines[l][0]], junctions[lines[l][1]]
    #         plt.plot([v1[0], v2[0]], [v1[1], v2[1]], c=cmap(i / len(vp)))
    # plt.show()
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

    candidates = set([i for i in range(len(lines)) if weights[i] > 0.05])
    clusters = []
    for i, j in combinations(candidates, 2):
        w = np.cross(normals[i], normals[j])
        if LA.norm(w) < 1e-4:
            continue
        w /= LA.norm(w)
        line_candidates = set(nearby_lines(w)) & candidates
        if len(line_candidates) > 3:
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
    # for i, (w, c) in enumerate(clusters):
    #     plt.figure()
    #     plt.xlim([-1, 1])
    #     plt.ylim([-1, 1])
    #     plt.scatter(w[0] / w[2], w[1] / w[2], c=cmap(i / len(clusters)))
    #     for l in c:
    #         v1, v2 = junctions[lines[l][0]], junctions[lines[l][1]]
    #         plt.plot([v1[0], v2[0]], [v1[1], v2[1]])

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
            if 1.8 <= K[0, 0] < 4 and cost < best_cost:
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

    # plt.figure(), plt.tight_layout()
    # for i, (w, c) in enumerate(vp):
    #     plt.xlim([-1, 1])
    #     plt.ylim([-1, 1])
    #     plt.scatter(w[0] / w[2], w[1] / w[2], c=cmap(i / len(vp)))
    #     for l in c:
    #         v1, v2 = junctions[lines[l][0]], junctions[lines[l][1]]
    #         plt.plot([v1[0], v2[0]], [v1[1], v2[1]], c=cmap(i / len(vp)))
    # plt.show()

    return vp


def vanish_point_refine(junctions, lines, vps, blacklist=[], total_iter=4, plot=False):
    vps = vps[:, :2] / vps[:, 2:]
    assignment = [[], [], []]

    for niter in range(total_iter):
        for i in range(3):
            if len(assignment[i]) <= 1:
                continue
            assignment[i].sort(key=lambda x: x[1])
            c = assignment[i][: math.ceil(len(assignment[i]) * 0.6)]
            c = [i for i, score in c]
            x1, y1 = junctions[lines[c, 0], :].T
            x2, y2 = junctions[lines[c, 1], :].T
            vps[i] = vp_from_lines(x1, y1, x2, y2)

        assignment = [[], [], []]
        for i, (a, b) in enumerate(lines):
            if i in blacklist:
                continue
            bestd = 1e100
            v = junctions[a] - junctions[b]
            for j in range(3):
                dist = abs(np.cross(v, vps[j] - junctions[b]))
                if dist < bestd:
                    bestd = dist
                    bestv = v
                    bestj = j
            assignment[bestj].append((i, bestd / LA.norm(bestv)))

    if plot:
        plt.figure(), plt.tight_layout()
        for i, (c, vp) in enumerate(zip(assignment, vps)):
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.scatter(vp[0], vp[1], c=cmap(i / len(vp)))
            for l, _ in c:
                v1, v2 = junctions[lines[l][0]], junctions[lines[l][1]]
                plt.plot([v1[0], v2[0]], [v1[1], v2[1]], c=cmap(i / len(vp)))
        plt.show()
    vps = np.c_[vps, np.ones(3)]

    for i in range(3):
        vps[i] /= LA.norm(vps[i])
    return [(vps[i], [i for i, score in assignment[i]]) for i in range(3)]


def lifting_from_vp(vp, invK, junctions, juncdepth, junctypes, lines, lambda_=0.01):
    vertices = np.c_[junctions, np.ones(len(junctions))] @ invK.T

    assert len(junctions) == len(junctypes)

    T = {}
    for i, (junc, typ) in enumerate(zip(junctions, junctypes)):
        if typ != 1:
            continue
        bestd = 1e9
        for j, (a, b) in enumerate(lines):
            if i in (a, b):
                continue
            ja, jb = junctions[a], junctions[b]
            vec = jb - ja
            u = (junc - ja) @ vec / (vec @ vec)
            if 1e-2 < u < 1 - 1e-2:
                d = LA.norm(ja + u * vec - junc)
                if d < bestd:
                    bestd = d
                    bestj = j
                    bestu = u
        if bestd < 1e9:
            T[i] = (bestj, bestu)

    # for a, b in lines:
    #     ja, jb = junctions[a], junctions[b]
    #     plt.plot([ja[0], jb[0]], [ja[1], jb[1]], color="green", zorder=0)
    # for i, (j, _) in T.items():
    #     plt.scatter(junctions[i][0], junctions[i][1], color="blue")
    #     ja, jb = junctions[lines[j][0]], junctions[lines[j][1]]
    #     plt.plot([ja[0], jb[0]], [ja[1], jb[1]], color="blue", zorder=10)
    # plt.show()

    def make_cvx_problem():
        depth = cvx.Variable(len(juncdepth))
        scale = cvx.Variable()
        objective = 0
        constraints = [depth >= 1, scale >= 0]
        for iw, (iline, u) in T.items():
            iu, iv = lines[iline]
            constraints.append(depth[iw] >= (1 - u) * depth[iu] + u * depth[iv])
        for w, c in vp:
            w = invK @ w
            w /= LA.norm(w)
            for l in c:
                i, j = lines[l]
                uv = depth[i] * vertices[i] - depth[j] * vertices[j]
                objective += cvx.norm(cvx.hstack(cross(uv, w)))
        for i in range(len(juncdepth)):
            if juncdepth[i] is None or juncdepth[i] == 0:
                continue
            objective += lambda_ * cvx.square(depth[i] - scale * juncdepth[i])
        problem = cvx.Problem(cvx.Minimize(objective), constraints)
        return problem, depth, scale

    problem, depth, scale = make_cvx_problem()
    problem.solve()
    # problem.solve(solver="SCS")

    return depth.value / scale.value


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
