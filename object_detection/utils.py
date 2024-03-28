import torch
import torchvision
import math
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def plot_pr_curve(px, py, ap, save_dir='pr_curve.jpg', names=()):
    # Precision-recall curve
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            plt.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(recall, precision)
    else:
        plt.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    plt.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(save_dir, dpi=250)


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="ciou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "diou":
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))
            outer_dig = torch.sum(torch.pow((c_br - c_tl), 2), 1)
            inter_dig = torch.sum(torch.pow((pred[:, :2] - target[:, :2]), 2), 1)
            diou = iou - inter_dig / outer_dig.clamp(min=1e-16)
            loss = 1 - diou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "ciou":
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))
            outer_dig = torch.sum(torch.pow((c_br - c_tl), 2), 1)
            inter_dig = torch.sum(torch.pow((pred[:, :2] - target[:, :2]), 2), 1)
            v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(pred[:, 2] / pred[:, 3]) - torch.atan(target[:, 2] / target[:, 3])), 2)
            S = 1 - iou
            alpha = v / (S + v)
            ciou = iou - inter_dig / outer_dig.clamp(min=1e-16) - alpha * v
            loss = 1 - ciou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, None] + area_b - area_i)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.5, classes=None, multi_label=False):

    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference

        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5, so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue

        # Batched NMS
        c = x[:, 5:6] * 224  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class, ensure that the detection bbox of different classes do not overlap)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        output[xi] = x[i]
    return output


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def compute_ap(recall, precision, method="interp"):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
        v5_metric: Assume maximum recall to be 1.0, as in YOLOv5, MMDetetion etc.
    # Returns
        Average precision, precision curve, recall curve
    """
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    if method == "interp":
        x = np.linspace(0, 1, 101)
        y = np.interp(x, mrec, mpre)# 101-point interp (COCO)
        ap = np.trapz(y, x)  # integrate
        # q = np.zeros((101,))
        # inds = np.searchsorted(mrec, x, side='left')
        # try:
        #     for ri, pi in enumerate(inds):
        #         q[ri] = mpre[pi]
        # except:
        #     pass
        # q_array = np.array(q)
        # ap = np.mean(q_array[q_array > -1])
    else:  # method == 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, method="interp", plot=False):
    """
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]

    px, py = np.linspace(0, 1, 1000), []
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 or n_l == 0:
            continue
        else:
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            recall = tpc / (n_l + 1e-16)
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

            precision = tpc / (tpc + fpc)
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], method)
                if j == 0:
                    py.append(np.interp(px, mrec, mpre))
    if plot:
        plot_pr_curve(px, py, ap, 'PR_curve.jpg', ["car", "pedestrian"])
    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap*100, f1[:, i], unique_classes.astype('int32')


def Compute_statistics(out, rel_labels, device, stats):
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    for si, pred in enumerate(out):
        labels = torch.cat((rel_labels[si, :, 0:4], rel_labels[si, :, 5:]), dim=1)
        nl = (labels.sum(dim=1) > 0).sum(dim=0)
        labels = labels[:nl]
        tcls = labels[:, 4].tolist() if nl else []  # target class
        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        predn = pred.clone()

        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool).to(device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 4]

            # target boxes
            tbox = xywh2xyxy(labels[:, 0:4])

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                    ious, index = ious.sort(descending=True)
                    i = i[index]
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv
                            if len(detected) == nl:
                                break

        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
    return stats


def Cumpute_map(stats, method="interp"):
    mp, mr, map50, map75, map = 0.0, 0.0, 0.0, 0.0, 0.0
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, method)
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
        # nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
    print("P         R         map0.5     map0.75    map0.5:0.95:0.05")
    # pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print("%02f  %02f  %02f%%  %02f%%  %02f%%" % (mp, mr, map50, map75, map))
    return map50, map75, map
    # Print results per class
    # for i, c in enumerate(ap_class):
    #     print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))


def sin_cos_pos_emb(seq_len, d, n=100):
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P


class BCEFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=0.75, reduction="None"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        pt = torch.sigmoid(input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.clamp(torch.log(pt), min=-100, max=0)\
               - (1 - alpha) * pt ** self.gamma * (1 - target) * torch.clamp(torch.log(1 - pt), min=-100, max=0)
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class my_BCELoss(torch.nn.Module):

    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction
        # self.softmax = torch.nn.Softmax(dim=1)
        self.CE = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, input, target):
        # pt = torch.sigmoid(input)
        # pt = self.softmax(input)
        # loss = - target[:, 0] * (torch.log(pt[:, 0])) - target[:, 1] * (torch.log(pt[:, 1]))
        loss = self.CE(input, target)
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss
