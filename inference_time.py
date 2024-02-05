import os
import cv2
import time
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn

from dass_det.models.yolox import YOLOX
from dass_det.models.yolo_head import YOLOXHead
from dass_det.models.yolo_head_stem import YOLOXHeadStem
from dass_det.models.yolo_pafpn import YOLOPAFPN
from dass_det.data.data_augment import ValTransform
from dass_det.utils import postprocess


data_path    = None # set a folder path where random images are given inside
model_dir    = None # set the model .pth path
mode         = 0
inc_postproc = True

repeat_cnt   = 100
warmup_cnt   = 10

# set to 0.33, 0.375 if model size is xs, set 1.33, 1.25 if model size is xl
depth, width = 0.33, 0.375 
resize_size  = (640, 640)
conf_thold   = 0.55
nms_thold    = 0.4

torch.backends.cudnn.benchmark = True

model = YOLOX(backbone=YOLOPAFPN(depth=depth, width=width),
              head_stem=YOLOXHeadStem(width=width),
              face_head=YOLOXHead(1, width=width),
              body_head=YOLOXHead(1, width=width))

d = torch.load(model_dir, map_location=torch.device('cpu'))

if "teacher_model" in d:
    model.load_state_dict(d["teacher_model"], strict=False)
else:
    model.load_state_dict(d["model"], strict=False)

model = model.eval().cuda()
transform = ValTransform()

for _ in range(warmup_cnt):

    series = os.listdir(data_path)
    serie = series[np.random.randint(len(series))]
    files = os.listdir(os.path.join(data_path, serie))
    file = files[np.random.randint(len(files))]
    filepath = os.path.join(data_path, serie, file)

    imgs = cv2.imread(filepath)
    h, w, c = imgs.shape
       
    imgs, labels = transform(imgs, None, resize_size)
    img_cu = torch.Tensor(imgs).unsqueeze(0).cuda()

    with torch.no_grad():
        face_preds, body_preds = model(img_cu, mode)
      
        if inc_postproc:
            
            if mode == 0:
                face_preds = postprocess(face_preds, 1, conf_thold, nms_thold)[0]
                body_preds = postprocess(body_preds, 1, conf_thold, nms_thold)[0]
            elif mode == 1:
                face_preds = postprocess(face_preds, 1, conf_thold, nms_thold)[0]
            else:
                body_preds = postprocess(body_preds, 1, conf_thold, nms_thold)[0]

total_secs = 0
for _ in tqdm(range(repeat_cnt)):

    series = os.listdir(data_path)
    serie = series[np.random.randint(len(series))]
    files = os.listdir(os.path.join(data_path, serie))
    file = files[np.random.randint(len(files))]
    filepath = os.path.join(data_path, serie, file)

    imgs = cv2.imread(filepath)
    h, w, c = imgs.shape
    imgs, labels = transform(imgs, None, resize_size)
    img_cu = torch.Tensor(imgs).unsqueeze(0).cuda()

    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():

        starter.record()
        face_preds, body_preds = model(img_cu, mode)
      
        if inc_postproc:
          
            if mode == 0:
                face_preds = postprocess(face_preds, 1, conf_thold, nms_thold)[0]
                body_preds = postprocess(body_preds, 1, conf_thold, nms_thold)[0]
            elif mode == 1:
                face_preds = postprocess(face_preds, 1, conf_thold, nms_thold)[0]
            else:
                body_preds = postprocess(body_preds, 1, conf_thold, nms_thold)[0]
      
        ender.record()
        torch.cuda.synchronize()
        
        total_secs += starter.elapsed_time(ender)
    
print('--> Inference Time: %.2f milliseconds.' % (total_secs / repeat_cnt))