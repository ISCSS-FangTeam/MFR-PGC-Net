import argparse
import numpy as np
import cv2
import six
import sys


def visual_refined_unary(refined_unary, rgb_img, refine=False):
    fg_img = np.uint8(255 * refined_unary[1])

    vis_result = cv2.applyColorMap(fg_img, cv2.COLORMAP_JET)
    # vis_result = cv2.addWeighted(rgb_img, 0.6, vis_result, 0.4, 0)  
    if refine:
        cv2.imwrite('./figs/unarys/refined_unary.png', vis_result)
    else:
        cv2.imwrite('./figs/unarys/unary.png', vis_result)
    sys.exit(0)


def visual_bam(Bg_unary, rgb_img, b_list):
    Bg_unary = Bg_unary[0].detach().cpu().numpy()
    normed_bam = np.ones_like(Bg_unary)
    for wmin, hmin, wmax, hmax, _ in b_list:
        unique = np.unique(Bg_unary[hmin:hmax, wmin:wmax])
        if len(unique) == 1:
            denom = Bg_unary[hmin:hmax, wmin:wmax].max() + 1e-12
            normed_bam[hmin:hmax, wmin:wmax] = Bg_unary[hmin:hmax, wmin:wmax] / denom
            continue
        mins = unique[1]
        denom = Bg_unary[hmin:hmax, wmin:wmax].max() + 1e-12
        indx_mask = normed_bam[hmin:hmax, wmin:wmax] == 0
        normed_bam[hmin:hmax, wmin:wmax] = (Bg_unary[hmin:hmax, wmin:wmax] - mins) / (denom - mins)
        normed_bam[hmin:hmax, wmin:wmax][np.where(normed_bam[hmin:hmax, wmin:wmax] < 0)] = 0
        # print(normed_bam[hmin:hmax,wmin:wmax].min(), normed_bam[hmin:hmax,wmin:wmax].max())
    bam = np.uint8(255 * normed_bam)
    bam = cv2.resize(bam, (80, 80), interpolation=cv2.INTER_CUBIC)
    bam = cv2.resize(bam, (512, 512), interpolation=cv2.INTER_CUBIC)
    vis_result = cv2.applyColorMap(bam, cv2.COLORMAP_JET)
    vis_result = cv2.addWeighted(rgb_img, 0.3, vis_result, 0.7, 0)  
    cv2.imwrite('./figs/bams/bam.png', vis_result)
    sys.exit(0)


def visual_cam(normed_cam_sum, rgb_img, b_list, index, fn):
    cam = (normed_cam_sum[0]).detach().cpu().numpy()
    normed_cam = np.zeros_like(cam)
    for wmin, hmin, wmax, hmax, _ in b_list:
        # unique = np.unique(cam[hmin:hmax,wmin:wmax])
        # if len(unique) == 1:
        #     denom = cam[hmin:hmax,wmin:wmax].max() + 1e-12
        #     normed_cam[hmin:hmax,wmin:wmax] = cam[hmin:hmax,wmin:wmax] / denom
        #     continue
        mins = cam[hmin:hmax, wmin:wmax].min()
        denom = cam[hmin:hmax, wmin:wmax].max() + 1e-12
        # indx_mask = normed_cam[hmin:hmax,wmin:wmax] == 0
        normed_cam[hmin:hmax, wmin:wmax] = (cam[hmin:hmax, wmin:wmax] - mins) / (denom - mins)
        # normed_cam[hmin:hmax,wmin:wmax][np.where(normed_cam[hmin:hmax,wmin:wmax]<0)] = 0
    normed_cam = np.uint8(255 * normed_cam)

    cam_img = cv2.applyColorMap(normed_cam, cv2.COLORMAP_JET)
    vis_result = cv2.addWeighted(rgb_img, 0.3, cam_img, 0.7, 0)  
    cv2.imwrite('./figs/cam_sum/cam_sum_' + fn +'_'+ index + '.png', vis_result)
    sys.exit(0)

def visual_cam_individual(normed_cam_sums, rgb_img, b_list, fn):
    for ind, normed_cam_sum in enumerate(normed_cam_sums):
        cam = (normed_cam_sum[0]).detach().cpu().numpy()
        normed_cam = np.zeros_like(cam)
        for wmin, hmin, wmax, hmax, _ in b_list:
            # unique = np.unique(cam[hmin:hmax,wmin:wmax])
            # if len(unique) == 1:
            #     denom = cam[hmin:hmax,wmin:wmax].max() + 1e-12
            #     normed_cam[hmin:hmax,wmin:wmax] = cam[hmin:hmax,wmin:wmax] / denom
            #     continue
            mins = cam[hmin:hmax, wmin:wmax].min()
            denom = cam[hmin:hmax, wmin:wmax].max() + 1e-12
            # indx_mask = normed_cam[hmin:hmax,wmin:wmax] == 0
            normed_cam[hmin:hmax, wmin:wmax] = (cam[hmin:hmax, wmin:wmax] - mins) / (denom - mins)
            # normed_cam[hmin:hmax,wmin:wmax][np.where(normed_cam[hmin:hmax,wmin:wmax]<0)] = 0
        normed_cam = np.uint8(255 * normed_cam)

        cam_img = cv2.applyColorMap(normed_cam, cv2.COLORMAP_JET)
        vis_result = cv2.addWeighted(rgb_img, 0.3, cam_img, 0.7, 0)  
        cv2.imwrite('./figs/cam_sum/' + fn +'_'+ str(ind) + '.png', vis_result)
    # sys.exit(0)



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = 0
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))

        if lb_max >= n_class:
            expanded_confusion = np.zeros((lb_max + 1, lb_max + 1), dtype=np.int64)
            # sys.exit(0)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels.
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) +
            pred_label[mask], minlength=n_class ** 2).reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
    return confusion


def get_mIOU(preds, gts, path_crf):
    
    # print(len(preds), len(gts))
    confusion = calc_semantic_segmentation_confusion(preds, gts)[:2, :2]  
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator
    # print("total images", n_img)
    # print(fp[0], fn[0])
    # print(np.mean(fp[1:]), np.mean(fn[1:]))

    precision = gtjresj / (fp * denominator + gtjresj)
    recall = gtjresj / (fn * denominator + gtjresj)
    F_score = 2 * (precision * recall) / (precision + recall)

    f = open("out.txt", "a")
    print('##################################################', file=f)
    print('{}'.format(path_crf), file=f)
    print('precision: [{:.4f},{:.4f}]'.
          format(precision[0], precision[1]), file=f)
    print('recall: [{:.4f},{:.4f}]'.
          format(recall[0], recall[1]), file=f)
    print('F_score: [{:.4f},{:.4f}]'.
          format(F_score[0], F_score[1]), file=f)
    print('iou: [{:.4f},{:.4f}], miou: {:.4f}'.format(iou[0], iou[1], np.nanmean(iou)), file=f)
    print('##################################################\n', file=f)
    f.close()

