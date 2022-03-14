#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

from dass_det.utils.boxes import xyxy2cxcywh


def draw_ellipse(img, center, radius_x, radius_y, color=(255, 255, 255)):
    
    num_letters = 5
    if radius_x == radius_y:
        num_letters = 10
    elif radius_x > radius_y:
        num_letters = 15
    
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?.,;:'"
    text = "".join(random.choices(letters, k=num_letters))
    
    angle, start, end = np.random.choice([0, 90]), 0, 360
    img = cv2.ellipse(img, center, (radius_x, radius_y), angle, start, end, color, thickness=-1)
    
    return img

def draw_rounded_rectangle(img, center, width, height, radius=None, color=(255, 255, 255)):
    
    num_letters = 5
    if width == height:
        num_letters = 10
    elif width > height:
        num_letters = 15
    
    
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?.,;:'"
    text = "".join(random.choices(letters, k=num_letters))

    if radius is None:
        radius = np.random.randint(0, min(width, height) // 3)
    
    # center rectangle
    img = cv2.rectangle(img, 
                        (center[0] - width//2 + radius, center[1] - height // 2 + radius),
                        (center[0] + width//2 - radius, center[1] + height // 2 - radius), 
                        color, thickness=-1)
    
    # top rectangle
    img = cv2.rectangle(img, 
                        (center[0] - width//2 + radius, center[1] - height // 2),
                        (center[0] + width//2 - radius, center[1] - height // 2 + radius), 
                        color, thickness=-1)
    
    # bottom rectangle
    img = cv2.rectangle(img, 
                        (center[0] - width//2 + radius, center[1] + height // 2 - radius),
                        (center[0] + width//2 - radius, center[1] + height // 2), 
                        color, thickness=-1)
    
    # left rectangle
    img = cv2.rectangle(img, 
                        (center[0] - width//2, center[1] - height // 2 + radius),
                        (center[0] - width//2 + radius, center[1] + height // 2 - radius), 
                        color, thickness=-1)
    
    # right rectangle
    img = cv2.rectangle(img, 
                        (center[0] + width//2 - radius, center[1] - height // 2 + radius),
                        (center[0] + width//2, center[1] + height // 2 - radius), 
                        color, thickness=-1)
    
    # topleft ellipse
    img = cv2.ellipse(img, 
                      (center[0] - width//2 + radius, center[1] - height // 2 + radius), 
                      (radius, radius), 90, 270, 0, color, thickness=-1)
    
    # topright ellipse
    img = cv2.ellipse(img, 
                      (center[0] + width//2 - radius, center[1] - height // 2 + radius), 
                      (radius, radius), 270, 0, 90, color, thickness=-1)
    
    # bottomleft ellipse
    img = cv2.ellipse(img, 
                      (center[0] - width//2 + radius, center[1] + height // 2 - radius), 
                      (radius, radius), 270, 180, 270, color, thickness=-1)
    
    # bottomright ellipse
    img = cv2.ellipse(img, 
                      (center[0] + width//2 - radius, center[1] + height // 2 - radius), 
                      (radius, radius), 270, 90, 180, color, thickness=-1)
    
    img = cv2.putText(img, text, (center[0] - width // 2, center[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
    
    return img 

def draw_speech_balloon(orig_img, ellipse_prob=0.7, max_balloons :int=2, add_noise :bool=True, horizontal_prob=0.75, downwards_prob=0.15):
    h, w, c = orig_img.shape
    
    img = np.zeros_like(orig_img)
    
    small_most_ratios = [0.5, 1.0, 0.3333, 0.6667, 0.25, 0.75, 0.6, 0.4, 0.4286, 
                         0.4444, 0.2, 0.5714, 0.2857, 0.8, 0.5556, 0.7143, 0.8333, 
                         0.375, 0.3636, 0.3077, 0.4545]
    
    big_most_ratios   = [1.1429, 1.2, 1.16, 1.33]
    
    bubble_colors     = [(255, 255, 255), (224, 224, 224), (153, 255, 255), 
                         (255, 229, 204), (204, 204, 255), (229, 255, 204)]
    
    num_balloons = 1 if max_balloons < 2 else np.random.randint(1, max_balloons+1)
    for _ in range(num_balloons):
        if np.random.rand() > horizontal_prob:
            ratio = np.random.choice(big_most_ratios)
        else:
            ratio = np.random.choice(small_most_ratios)
        
        if ratio < 0.35:
            bh = np.random.randint(min(40, 2*h // 3), max(40, 2*h // 3))
            bw = min(int(bh * ratio), w)
        elif ratio <= 1:
            bh = np.random.randint(min(40, h // 3), max(40, h // 3))
            bw = min(int(bh * ratio), w)
        else:
            bw = np.random.randint(min(40, w // 3), max(40, w // 3))
            bh = min(int(bw / ratio), h)
        
        cx, cy = np.random.randint(int(0.2*w), int(0.81*w)), np.random.randint(int(0.2*h), int(0.4*h))
        
        if np.random.rand() <= downwards_prob:
            cy = h - cy
        
        if np.random.rand() < 0.4:
            color = bubble_colors[0]
        elif np.random.rand() < 0.6:
            color = bubble_colors[1]
        else:
            color = random.choice(bubble_colors[2:])
        
        if np.random.rand() > ellipse_prob:
            img = draw_rounded_rectangle(img, (cx, cy), bw, bh, color=color)
        else:
            img = draw_ellipse(img, (cx, cy), bw // 2, bh // 2, color=color)
        
    
    if add_noise:
        noise_ratio = 25
        noise_xs = np.random.randint(w, size=(int(max(w, h)*noise_ratio)))
        noise_ys = np.random.randint(h, size=(int(max(w, h)*noise_ratio)))
        img[noise_ys, noise_xs, :] = 0
    
    orig_img = np.maximum(orig_img, img)
    
    return orig_img

def convert_grayscale(img):
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    return gray_img

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates

# ----------------------------------------------------------------------------------------------

def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )

def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
    perp_rotate_prob=0.0,
):
    twidth, theight = target_size

    # Rotation and Scale
    if random.random() < perp_rotate_prob:
        angle = random.choice([90, -90])
    else:
        angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D((twidth // 2, theight // 2), angle, scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]
    
    return M, scale

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] += translation_x
    M[1, 2] += translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
    perp_rotate_prob=0.0,
):
    
    target_size = img.shape[1::-1]
    
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear, perp_rotate_prob)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114), flags=cv2.INTER_LINEAR)

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets


def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def _vertical_mirror(image, boxes, prob=0.5):
    height, width, _ = image.shape
    if random.random() < prob:
        image = image[::-1, ...]
        boxes[:, 1::2] = height - boxes[:, 3::-2]
    return image, boxes


def random_crop(image, min_ratio=0.5):
    assert min_ratio <= 1
    
    h, w     = image.shape[:2]
    min_len  = min(h, w)
    side_len = np.random.randint(int(min_len * min_ratio)-1, min_len)
    x_start  = np.random.randint(0, w-side_len)
    y_start  = np.random.randint(0, h-side_len)
    
    vals = {
        "length": side_len,
        "x": x_start,
        "y": y_start
    }
    
    image = image[y_start:y_start+side_len, x_start:x_start+side_len, ...]
    return image, vals


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, vertical_flip_prob=0.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        
        if targets is None:
            targets = np.zeros((2, 5), dtype=np.float32)
        
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            
            if random.random() < 0.95:
                augment_hsv(image)
            else:
                image = convert_grayscale(image)
                
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        image_t, boxes = _vertical_mirror(image_t, boxes, self.vertical_flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels 
    
class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))
    

class StrongTransform:
    
    def __init__(self, swap=(2, 0, 1), flip_prob=0.5, hsv_prob=0.7, crop_prob=1.0, 
                 min_crop_ratio=0.5, gaussian_noise_prob=0.15, vertical_flip_prob=0.00):
        self.swap = swap
        self.flip_prob = flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.hsv_prob = hsv_prob
        self.crop_prob = crop_prob
        self.min_crop_ratio = min_crop_ratio
        self.gaussian_noise_prob = gaussian_noise_prob

    def __call__(self, img, input_size):
        
        changes = {"ratio": 1, "hflip": False, "vflip": False, "crop":None}
        
        if random.random() < self.hsv_prob:
            augment_hsv(img)
        
        if random.random() < self.flip_prob:
            img = img[:, ::-1, ...]
            changes["hflip"] = True
        
        if random.random() < self.vertical_flip_prob:
            img = img[::-1, ...]
            changes["vflip"] = True
            
        if random.random() < self.crop_prob:
            img, crop_vals = random_crop(img, min_ratio=self.min_crop_ratio)
            changes["crop"] = crop_vals
        
        if random.random() < self.gaussian_noise_prob:
            gaussian_noise = np.random.rand(*img.shape[:-1], 1)
            img = img * (gaussian_noise < (1-self.gaussian_noise_prob))
        
        img, ratio = preproc(img, input_size, self.swap)
        changes["ratio"] = ratio
        
        return img, changes

    
class WeakTransform:

    def __init__(self, swap=(2, 0, 1), flip_prob=0.5):
        self.swap = swap
        self.flip_prob = flip_prob

        
    def __call__(self, img, input_size):
        
        changes = {"ratio": 1, "hflip": False}
        
        if random.random() < self.flip_prob:
            img = img[:, ::-1]
            changes["hflip"] = True
        
        img, ratio = preproc(img, input_size, self.swap)
        changes["ratio"] = ratio
        
        return img, changes
