import os
import cv2
import copy
import random

import numpy as np
from tqdm import tqdm

from PIL import Image

import xmltodict

from collections import OrderedDict
from loguru import logger


from .datasets_wrapper import Dataset


class EBDthequeDataset(Dataset):
    """
    eBDtheque dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        train=True,
        img_size=(416, 416),
        preproc=None,
        cache=False, # no cache is supported currently
    ):
        super().__init__(img_size)
        self.class_ids = [0, 1]
        self.class_dict = {"frame": self.class_ids[0], "body": self.class_ids[1]}
        self.img_size = img_size
        self.preproc = preproc   
        self.files, self.annotations = self.load_annotations(data_dir, train)
        
        
    def __len__(self):
        return len(self.files)
    
    
    def pull_item(self, index):
        img, img_info = self.load_image(index)
        res = self.load_anno(index)
        return img, res.copy(), img_info, np.array([index])
    
    
    def __getitem__(self, index):
        img, target, img_info, ids = self.pull_item(index)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, ids
    
    
    def load_anno(self, index):
        # given an index, it loads the annotations of the file at that index
        file = self.files[index]
        body_annots = self.annotations[file]["body"]
        frame_annots = self.annotations[file]["frame"]
        anno = np.zeros((len(body_annots) + len(frame_annots), 5))

        for idx, ann in enumerate(body_annots):
            anno[idx,:4] = ann
            anno[idx,4] = self.class_dict["body"]
        
        for idx, ann in enumerate(frame_annots):
            anno[idx,:4] = ann
            anno[idx,4] = self.class_dict["frame"]
        
        return anno
    
    
    def load_resized_img(self, index):
        img, img_info = self.load_image(index)
        
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img, img_info

    def load_image(self, index):
        img_path = self.files[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert img is not None
        h, w, c = img.shape
        return img, [h, w]
    
    
    def load_annotations(self, edb_paths, train :bool):
        # gven a path and partition, it loads all the image paths and annots in that partition
        files = []
        boxes = {}
        
        img_path, annot_path = edb_paths["imgs"], edb_paths["labels"]

        for page in os.listdir(annot_path):
            f = open(os.path.join(annot_path, page), "r")
            book_annots = xmltodict.parse(f.read())
            f.close()
            
            k = os.path.join(img_path, page[:-4] + ".jpg")
            files.append(k)
            boxes[k] = {"frame": [], "body": []}
            
            for idx in [1, 4]:
                for d in book_annots["svg"]["svg"][idx]["polygon"]:
                    try:
                        pts = d["@points"]
                    except:
                        pts = book_annots["svg"]["svg"][idx]["polygon"]["@points"]
                    
                    pts = pts.split(" ")
                    
                    x1, y1, x2, y2 = 1000000, 10000000, -1, -1
                    for p in pts:
                        x, y = p.split(",")
                        x, y = int(x), int(y)
                        x1, y1 = min(x, x1), min(y, y1)
                        x2, y2 = max(x, x2), max(y, y2)
                    
                    if idx == 1:
                        boxes[k]["frame"].append([x1, y1, x2, y2])
                    elif idx == 4:
                        boxes[k]["body"].append([x1, y1, x2, y2])

        return files, boxes