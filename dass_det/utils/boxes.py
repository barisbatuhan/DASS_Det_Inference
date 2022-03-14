#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "intersect",
    "check_center_boxa_in_boxb",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cxcywh2xyxy",
    "xywh2xyxy"
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]

def check_center_boxa_in_boxb(box_a, box_b, cxcy_format=False):
    """
    Checks if the center of box_a is in box_b.
    Return:
      (tensor) int tensor of zeros and ones.
    """
    
    A = box_a.size(0)
    B = box_b.size(0)
    
    if cxcy_format:
        centers = box_a[:,:2].unsqueeze(1).expand(A, B, 2)
        mins = (box_b[:,:2] - box_b[:,2:] / 2).unsqueeze(0).expand(A, B, 2)
        maxs = (box_b[:,:2] + box_b[:,2:] / 2).unsqueeze(0).expand(A, B, 2)
    else:
        centers = ((box_a[:,2:] + box_a[:,:2]) / 2).unsqueeze(1).expand(A, B, 2)
        mins = box_b[:,:2].unsqueeze(0).expand(A, B, 2)
        maxs = box_b[:,2:].unsqueeze(0).expand(A, B, 2)
    
    min_x = centers[:,:,0] >= mins[:,:,0]
    min_y = centers[:,:,1] >= mins[:,:,1]
    max_x = centers[:,:,0] <= maxs[:,:,0]
    max_y = centers[:,:,1] <= maxs[:,:,1]
    
    final = min_x * max_x * min_y * max_y
    return final
    

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False, helper_prediction=None):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        
        if helper_prediction is not None:
            helper_inst = helper_prediction[i]
        else:
            helper_inst = None
            
        if helper_inst is not None:

            tops = torch.argsort(image_pred[:,4], descending=True)
            image_pred = image_pred[tops,:]

            inters = intersect(helper_inst[:,:4], image_pred[:,:4])
            farea = helper_inst[:,2:4] - helper_inst[:,:2]
            farea = farea[:,0] * farea[:,1]
            inters = inters / farea.unsqueeze(-1)
            fbox_axes, bbox_axes = torch.where(inters > 0.95)

        detections = image_pred[:, :5]
        
        if  helper_inst is not None:
            res = torch.zeros(helper_inst.shape[0], 2)
            for j in range(fbox_axes.shape[0]):
                x, y = fbox_axes[j], bbox_axes[j]
                if detections[y, 4] > res[x, 0]:
                    res[x, 0] = detections[y, 4]
                    res[x, 1] = y
            
            for j in range(res.shape[0]):
                y = res[j, 1].long()
                if detections[y, 4] < helper_inst[j, 4]:
                    detections[y, 4] = (detections[y, 4] + helper_inst[j, 4]) / 2
            
        conf_mask = (detections[:, 4] >= conf_thre).squeeze() 
        detections = detections[conf_mask]

        if not detections.size(0):
            continue
            
        nms_out_index = torchvision.ops.nms(
            detections[:, :4],
            detections[:, 4],
            nms_thre,
        )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


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
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

def cxcywh2xyxy(bboxes):
    bboxes[:, 0] -= bboxes[:, 2] / 2
    bboxes[:, 1] -= bboxes[:, 3] / 2
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return bboxes
    
def xywh2xyxy(bboxes):
    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
    return bboxes
