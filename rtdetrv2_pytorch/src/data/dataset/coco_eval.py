"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from ...misc import dist_utils
from ...core import register

__all__ = ['CocoEvaluator',]


# --- Optional fast evaluation (faster-coco-eval) -----------------------------
# Enabled with FAST_COCOEVAL=1. Gives the same results as pycocotools
# (difference < 1e-15 in stats and precision) but evaluates in C++. If the package
# is not installed, it falls back to pycocotools with a warning.
_FAST_COCOEVAL = os.environ.get("FAST_COCOEVAL", "0") == "1"
if _FAST_COCOEVAL:
    try:
        from faster_coco_eval import COCOeval_faster
    except ImportError:
        print("FAST_COCOEVAL=1 but faster-coco-eval is not installed: using pycocotools")
        _FAST_COCOEVAL = False


def _fast_evaluate(coco_gt, results, img_ids, iou_type):
    """Evaluate everything at once with faster-coco-eval and return the COCOeval object."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            if results:
                coco_eval = COCOeval_faster(coco_gt, COCO.loadRes(coco_gt, results), iou_type)
            else:  # no detections: pycocotools, as in the normal path
                coco_eval = COCOeval(coco_gt, COCO(), iouType=iou_type)
            coco_eval.params.imgIds = list(img_ids)
            coco_eval.evaluate()
            coco_eval.accumulate()
    return coco_eval



@register()
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)

        if 'info' not in coco_gt.dataset:
            coco_gt.dataset['info'] = []
        
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self._fast_results = {}
    
    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval(self.coco_gt, iouType=iou_type)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}
        self._fast_results = {}
    
    
    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            if _FAST_COCOEVAL:  # only accumulate; everything is evaluated in accumulate()
                self._fast_results.setdefault(iou_type, []).extend(results)
                continue

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        if _FAST_COCOEVAL:
            for iou_type in self.iou_types:
                gathered = dist_utils.all_gather(self._fast_results.get(iou_type, []))
                self._fast_results[iou_type] = [r for part in gathered for r in part]
            self.img_ids = sorted({int(i) for part in dist_utils.all_gather(self.img_ids) for i in part})
            return
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        if _FAST_COCOEVAL:
            for iou_type in self.iou_types:
                self.coco_eval[iou_type] = _fast_evaluate(
                    self.coco_gt, self._fast_results.get(iou_type, []), self.img_ids, iou_type
                )
            return
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

        # Store per-class metrics for logging
        self.per_class_metrics = self.compute_per_class_metrics('bbox')

    def compute_per_class_metrics(self, iou_type='bbox'):
        """
        Compute per-class Average Precision and Average Recall metrics.

        Args:
            iou_type (str): Type of IoU metric ('bbox', 'segm', 'keypoints')

        Returns:
            dict: {category_id: {'AP': ap_value, 'AP50': ap50_value, 'AP75': ap75_value, 'AR': ar_value}}
        """
        if iou_type not in self.coco_eval:
            return {}

        coco_eval = self.coco_eval[iou_type]

        # Ensure evaluate and accumulate have been called
        if not hasattr(coco_eval, 'eval') or coco_eval.eval is None:
            return {}

        precision = coco_eval.eval['precision']  # [T, R, K, A, M]
        # T: iou thresholds (10 values from 0.50:0.95)
        # R: recall thresholds (101 values from 0:1)
        # K: categories (num_classes)
        # A: area ranges (4: all, small, medium, large)
        # M: max detections (3: 1, 10, 100)

        recall = coco_eval.eval['recall']  # [T, K, A, M]

        per_class_metrics = {}

        for idx, cat_id in enumerate(coco_eval.params.catIds):
            # AP @ IoU=0.50:0.95 (average over all IoU thresholds, area=all, maxDets=100)
            ap_all = precision[:, :, idx, 0, 2]
            ap = ap_all[ap_all > -1].mean() if (ap_all > -1).any() else 0.0

            # AP @ IoU=0.50 (first IoU threshold)
            ap50_all = precision[0, :, idx, 0, 2]
            ap50 = ap50_all[ap50_all > -1].mean() if (ap50_all > -1).any() else 0.0

            # AP @ IoU=0.75 (6th IoU threshold: 0.50 + 5*0.05 = 0.75)
            ap75_all = precision[5, :, idx, 0, 2]
            ap75 = ap75_all[ap75_all > -1].mean() if (ap75_all > -1).any() else 0.0

            # AR @ maxDets=100 (average over all IoU thresholds, area=all)
            ar_all = recall[:, idx, 0, 2]
            ar = ar_all[ar_all > -1].mean() if (ar_all > -1).any() else 0.0

            per_class_metrics[cat_id] = {
                'AP': float(ap),
                'AP50': float(ap50),
                'AP75': float(ap75),
                'AR': float(ar),
            }

        return per_class_metrics

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


# import io
# from contextlib import redirect_stdout
# def evaluate(imgs):
#     with redirect_stdout(io.StringIO()):
#         imgs.evaluate()
#     return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))


def evaluate(self):
    """
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    """
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################

