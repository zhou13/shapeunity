import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from wireframe.utils import argsort2d

DX = [0, 0, 1, -1, 1, 1, -1, -1]
DY = [1, -1, 0, 0, 1, -1, 1, -1]


def ap(tp, fp, npos):
    recall = tp / npos
    precision = tp / np.maximum(tp + fp, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[i + 1] - recall[i]) * precision[i + 1])


def eval_depth(pred, pred_depth, gt, gt_depth, max_distance):
    confidence = pred[:, -1]
    sorted_ind = np.argsort(-confidence)
    nd = len(pred)
    pred = pred[sorted_ind, :-1]
    pred_depth = pred_depth[sorted_ind]
    d = np.sqrt(np.sum(pred ** 2, 1)[:, None] + np.sum(gt ** 2, 1)[None, :] - 2 * pred @ gt.T)
    choice = np.argmin(d, 1)
    hit = np.zeros(len(gt), np.bool)
    dist = np.min(d, 1)
    depth_diff = np.zeros(len(pred))

    for i in range(nd):
        if dist[i] < max_distance and not hit[choice[i]]:
            hit[choice[i]] = True
            a = np.maximum(-pred_depth[i], 1e-5)
            b = -gt_depth[choice[i]]
            depth_diff[i] = np.log(a) - np.log(b)

    n = np.maximum(np.sum(hit), 1)
    rst = np.sum(depth_diff @ depth_diff.T) / n - np.sum(depth_diff) * np.sum(depth_diff) / (n * n)

    return rst


def mAP_jlist(v0, v1, max_distance, im_ids, weight,
              pred_dirs=None, gt_dirs=None, weight_dirs=None):
    if len(v0) == 0:
        return 0

    # whether simultaneously evaluate direction prediction
    eval_dir = False
    if pred_dirs is not None:
        assert (gt_dirs is not None) and (weight_dirs is not None)
        eval_dir = True
        weight_dir_sum = sum([np.sum(j) for j in weight_dirs])
        gt_num = sum([np.sum(len(j)) for j in weight_dirs])
        weight_dirs = [_ / weight_dir_sum * gt_num for _ in weight_dirs]

    v0 = np.array(v0)
    v1 = np.array(v1)
    weight_sum = sum([np.sum(j) for j in weight])
    gt_num = sum([np.sum(len(j)) for j in weight])
    weight = [_ / weight_sum * gt_num for _ in weight]

    confidence = v0[:, -1]
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    v0 = v0[sorted_ind, :]
    im_ids = im_ids[sorted_ind]

    nd = len(im_ids)
    tp, fp = np.zeros(nd, dtype=np.float), np.zeros(nd, dtype=np.float)
    hit = [[False for _ in j] for j in v1]

    if eval_dir:
        pred_dirs = pred_dirs[sorted_ind]
        tp_dir, fp_dir = np.zeros(nd, dtype=np.float), np.zeros(nd, dtype=np.float)
        hit_dir = [[False for _ in j] for j in v1]

    # go down dets and mark TPs and FPs
    for i in range(nd):
        gt_juns = v1[im_ids[i]]
        pred_juns = v0[i][:-1]
        if len(gt_juns) > 0:
            # compute overlaps
            dists = np.linalg.norm((pred_juns[None, :] - gt_juns), axis=1)
            choice = np.argmin(dists)
            dist = np.min(dists)
            if dist < max_distance and not hit[im_ids[i]][choice]:
                tp[i] = weight[im_ids[i]][choice]
                hit[im_ids[i]][choice] = True
                # theta is correct only when junction is correct first
                if eval_dir:
                    gt_dir = gt_dirs[im_ids[i]][choice]
                    pred_dir = pred_dirs[i]
                    d_theta = np.fmod(gt_dir - pred_dir, 2 * np.pi)
                    d_theta = d_theta + 2 * np.pi if d_theta < 0 else d_theta
                    d_theta = np.minimum(np.abs(d_theta),
                                         np.abs(2 * np.pi - d_theta))
                    if d_theta < 2 * np.pi / 48.0 and \
                            not hit_dir[im_ids[i]][choice]:
                        tp_dir[i] = weight_dirs[im_ids[i]][choice]
                        hit_dir[im_ids[i]][choice] = True
                    else:
                        fp_dir[i] = 1
            else:
                fp[i] = 1
                if eval_dir:
                    fp_dir[i] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if eval_dir:
        tp_dir = np.cumsum(tp_dir)
        fp_dir = np.cumsum(fp_dir)
        return ap(tp, fp, gt_num), ap(tp_dir, fp_dir, gt_num)
    else:
        return ap(tp, fp, gt_num)


def nms_junction(heatmap, delta=1):
    heatmap = heatmap.copy()
    disable = np.zeros_like(heatmap, dtype=np.bool)
    for x, y in argsort2d(heatmap):
        for dx, dy in zip(DX, DY):
            xp, yp = x + dx, y + dy
            if not (0 <= xp < heatmap.shape[0] and 0 <= yp < heatmap.shape[1]):
                continue
            if heatmap[x, y] >= heatmap[xp, yp]:
                disable[xp, yp] = True
    heatmap[disable] = 0
    return heatmap


def ap_jheatmap(pred, truth, distances, im_ids, weight,
                pred_dir=None, gt_dir=None, weight_dir=None):
    # note the distance is junction prediction requirement
    # theta requirement is always fixed for now
    if pred_dir is not None:
        assert (gt_dir is not None) and (weight_dir is not None)
        ap_jt, ap_dirt = [], []
        for d in distances:
            j, d = mAP_jlist(pred, truth, d, im_ids, weight,
                             pred_dir, gt_dir, weight_dir)
            ap_jt.append(j)
            ap_dirt.append(d)
        return sum(ap_jt) / len(ap_jt) * 100, \
               sum(ap_dirt) / len(ap_dirt) * 100
    else:
        return sum(mAP_jlist(pred, truth, d, im_ids, weight)
                   for d in distances) / len(distances) * 100


def post_jheatmap(heatmap, offset=None, delta=1, dir_map=None, jdep_map=None):
    # heatmap = nms_junction(heatmap, delta=delta)
    # only select the best 1000 junctions for efficiency
    v0 = argsort2d(-heatmap)[:1000]
    confidence = -np.sort(-heatmap.ravel())[:1000]
    keep_id = np.where(confidence >= 1e-2)[0]
    if len(keep_id) == 0:
        return np.zeros((0, 3))

    v0 = v0[keep_id]
    confidence = confidence[keep_id]
    if offset is not None:
        v0 = np.array([v + offset[:, v[0], v[1]] for v in v0])
    v0 = np.hstack((v0, confidence[:, np.newaxis]))
    if dir_map is not None:
        assert offset is None
        # take the theta corresponding to v0
        # currently only support T direction so
        if len(dir_map.shape) == 2:
            dir = np.array([dir_map[int(v[0]), int(v[1])] for v in v0])
        else:
            raise NotImplementedError
        return v0, dir

    if jdep_map is not None:
        if len(jdep_map.shape) == 2:
            jdep = np.array([jdep_map[int(v[0]), int(v[1])] for v in v0])
        else:
            raise NotImplementedError
        return v0, jdep

    return v0


def get_confusion_mat(pred, gt):
    index = gt * 2 + pred
    label_count = np.bincount(index.reshape(-1).astype(np.int32))
    confusion_mat = np.zeros((2, 2))
    for i_label in range(2):
        for j_label in range(2):
            cur_index = i_label * 2 + j_label
            if cur_index < len(label_count):
                confusion_mat[i_label, j_label] = label_count[cur_index]
    return confusion_mat


def iou_line(confusion_mat, target_cls):
    pos = confusion_mat.sum(1)
    res = confusion_mat.sum(0)
    tp = np.diag(confusion_mat)
    iou = (tp / np.maximum(1.0, pos + res - tp))
    line_iou = iou[target_cls] * 100
    return line_iou


# def main():
#     a = np.random.randn(100, 2)
#     b = np.random.randn(100, 2)
#     c = np.concatenate([a, b], axis=0)
#
#     print("total match", mAP_jlist(a, a, 0.01))
#     print("half match", mAP_jlist(a[:50], a, 0.01))
#     print("ordered", mAP_jlist(c, a, 0.01))
#     np.random.shuffle(c)
#     print("disordered", mAP_jlist(c, a, 0.01))
#     print("no match", mAP_jlist(b, a, 0.01))


# if __name__ == "__main__":
#     main()
