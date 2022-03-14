#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.


import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tqdm import tqdm
import numpy as np

from chainercv.evaluations import eval_detection_voc, calc_detection_voc_prec_rec

import torch

from dass_det.utils import (
    postprocess,
    xyxy2xywh
)


class ComicEvaluator:
    """
    Manga109 AP Evaluation class.
    """

    def __init__(
        self, dataloader, img_size, confthre, nmsthre, num_classes, 
        use_real_size=False, verbose=False
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.verbose = verbose
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.use_real_size = use_real_size # if set true, then self.img_size will not be used

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        use_07_metric=False,
        select_index :int=-1,
        mode=1,
        use_helper_pred=False,
    ):
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        pred_bboxes, pred_labels, pred_scores = [], [], []
        gt_bboxes, gt_labels = [], []
        
        dl_iter = iter(self.dataloader)
        
        for cur_iter in progress_bar(range(len(self.dataloader))):
            
            if self.use_real_size:
                file_ids = [[cur_iter]]
                imgs, (h, w) = self.dataloader.dataset.load_image(cur_iter)
                resize_size = ((h // 32) * 32, (w // 32) * 32)   
                self.img_size = resize_size
                info_imgs = [[h], [w]]  
                
                imgs, labels = self.dataloader.dataset.preproc(imgs, None, resize_size)
                imgs = torch.from_numpy(imgs).unsqueeze(0)
            
            else:
                imgs, _, info_imgs, file_ids = next(dl_iter)
            
            with torch.no_grad():

                imgs = imgs.type(tensor_type)


                if mode == 1:
                    outputs, _ = model(imgs, mode=mode)
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())
                           
                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                
                elif mode == 2 and use_helper_pred:
                    foutputs, outputs = model(imgs, mode=0)
                    if decoder is not None:
                        foutputs = decoder(foutputs, dtype=outputs.type())
                        outputs = decoder(outputs, dtype=outputs.type())
                    
                    foutputs = postprocess(foutputs, self.num_classes, 0.7, 0.4)
                    if foutputs is not None:
                        outputs = postprocess(outputs, self.num_classes, self.confthre, 
                                              self.nmsthre, helper_prediction=foutputs)
                    else:
                        outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                    
                elif mode == 2:
                    
                    _, outputs = model(imgs, mode=mode)
                    
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())
    
                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
            pred_box, pred_label, pred_score, gt_box, gt_label = self.extract_components(outputs,
                                                                                         info_imgs, 
                                                                                         file_ids,
                                                                                         select_index)
            
            pred_bboxes.extend(pred_box)
            pred_labels.extend(pred_label)
            pred_scores.extend(pred_score)
            gt_bboxes.extend(gt_box)
            gt_labels.extend(gt_label)
            
            
        result = eval_detection_voc(pred_bboxes, 
                                    pred_labels, 
                                    pred_scores,
                                    gt_bboxes, 
                                    gt_labels, 
                                    use_07_metric=use_07_metric)
        
        pre_rec_results = calc_detection_voc_prec_rec(pred_bboxes, 
                                                      pred_labels, 
                                                      pred_scores,
                                                      gt_bboxes, 
                                                      gt_labels)
        
        aps = result['ap']
        aps = aps[~np.isnan(aps)]
        
        if self.verbose:
            print(' ---> mAP: {:f}'.format(100.0 * result['map']))
        
        
        def get_key(my_dict, val):
            for key, value in my_dict.items():
                 if val == value:
                        return key.capitalize()
 
            return "Not Found"
        
        print_text = "AP Values | "
        for i in range(len(aps)):
            print_text += f' {get_key(self.dataloader.dataset.class_dict, i)}: {100.0 * aps[i]}'
        
        if self.verbose:
            print(print_text)
        
        result["precision"], result["recall"] = [], []
        
        for ev_idx in range(len(pre_rec_results[0])):
            if pre_rec_results[0][ev_idx] is not None and len(pre_rec_results[0][ev_idx]) > 1:
                result["precision"].append(pre_rec_results[0][ev_idx].mean())
                result["recall"].append(pre_rec_results[1][ev_idx].mean())
            else:
                result["precision"].append(0.0)
                result["recall"].append(0.0)
        
        if self.verbose:
            print("Precision:", result["precision"])
            print("Recall:", result["recall"])

        return result


    def extract_components(self, outputs, info_imgs, ids, select_index :int):
        
        pred_bboxes, pred_labels, pred_scores = [], [], []
        gt_bboxes, gt_labels = [], []
        
        if outputs is None:
            outputs = [None]*len(ids)
        
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is not None:
                output = output.cpu()
                bbox = output[:, 0:4]
                
                # preprocessing: resize
                scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
                bbox /= scale
    
                if output.shape[0] > 0:
                    cls = torch.zeros(output.shape[0]).fill_(max(0, select_index))
                else:
                    cls = torch.zeros(0)
                        
                score = output[:, 4]
                
            else:
                bbox = torch.zeros(0, 4)
                score = torch.zeros(0)
                cls = torch.zeros(0)
            
            gt_annot = self.dataloader.dataset.load_anno(img_id[0])
            gt_box = gt_annot[:,:4]
            gt_label = gt_annot[:,-1]
            
            pred_bboxes.append(bbox.cpu().numpy().astype(int))
            pred_labels.append(cls.cpu().numpy())
            pred_scores.append(score.cpu().numpy())
            gt_bboxes.append(gt_box.astype(int))
            gt_labels.append(gt_label)
            
        return pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels
