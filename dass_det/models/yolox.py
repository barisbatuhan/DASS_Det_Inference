#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_head_stem import YOLOXHeadStem
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head_stem=None, face_head=None, body_head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if face_head is None:
            face_head = YOLOXHead(1)
        if body_head is None:
            body_head = YOLOXHead(1)
        if head_stem is None:
            head_stem = YOLOXHeadStem()

        self.backbone = backbone
        self.face_head = face_head
        self.body_head = body_head
        self.head_stem = head_stem

    # mode = 0 for both heads, 1 for face, 2 for body
    def forward(self, x, mode=0):
        
        fpn_outs = self.backbone(x)
        fpn_outs = self.head_stem(fpn_outs)
        
        assert mode in [0, 1, 2]
            
        foutputs, boutputs = None, None
        
        if mode in [0, 1]:
            foutputs = self.face_head(fpn_outs)
        if mode in [0, 2]:
            boutputs = self.body_head(fpn_outs)
        
        return foutputs, boutputs
