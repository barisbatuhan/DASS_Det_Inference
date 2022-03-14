import os
from collections import OrderedDict
from loguru import logger

import cv2
import numpy as np

from .datasets_wrapper import Dataset


class DCM772Dataset(Dataset):
    """
    DCM 772 dataset class.
    """
    def __init__(
        self,
        data_dir=None,
        train=True,
        img_size=(416, 416),
        preproc=None,
        cache=False, # no cache is supported currently
        include_animals :bool=True,
        include_back_chars :bool=True,
    ):
        super().__init__(img_size)
        self.class_ids = [0, 1, 2]
        self.class_dict = {"frame": self.class_ids[0], 
                            "face" : self.class_ids[1], 
                            "body" : self.class_ids[2]}
        
        # 5 is animal body, 6 is backgound person
        self.annot_cls_maps = {1: "body", 7:"face", 8:"frame"}
        if include_animals:
            self.annot_cls_maps[5] = "body"
        if include_back_chars:
             self.annot_cls_maps[6] = "body"
        
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
        annots = self.annotations[file]
        frames, faces, bodies = annots["frame"], annots["face"], annots["body"]
        total_len = len(frames) + len(faces) + len(bodies)
        anno = np.zeros((total_len, 5))
        idx = 0
        for frame in frames:
            anno[idx,:4] = frame
            anno[idx,4] = self.class_dict["frame"]
            idx += 1
        for face in faces:
            anno[idx,:4] = face
            anno[idx,4] = self.class_dict["face"]
            idx += 1
        for body in bodies:
            anno[idx,:4] = body
            anno[idx,4] = self.class_dict["body"]
            idx += 1
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
    
    
    def load_annotations(self, paths, train :bool):
        # given a path and partition, it loads all the image paths and annots in that partition
        
        files = []
        boxes = {}
        
        if type(paths) == str:
            dcm_path = paths
        else:
            dcm_path = paths["dcm772"]
            
        img_path    = os.path.join(dcm_path, "images")
        labels_path = os.path.join(dcm_path, "groundtruth")
        
        file_list = os.path.join(dcm_path, "train.txt") if train else os.path.join(dcm_path, "test.txt")
        f = open(file_list, "r")
        group = f.readlines()
        f.close()
        
        for file in group:
            # changes in file
            if len(file) < 2:
                break
            elif file[-1] == "\n":
                file = file[:-1]
            elif file[-4:].lower() in [".jpg", ".txt", ".png"]:
                file = file[:-4]
            
            annot_file = os.path.join(labels_path, file + ".txt")
            img_file = os.path.join(img_path, file + ".jpg")
            
            boxes[img_file] = {"frame":[], "face":[], "body":[]}
            
            with open(annot_file, "r") as f:
                lines = f.readlines()
            
            for line in lines:
                if len(line) < 2:
                    continue
                elif line[-1] == "\n":
                    line = line[:-1]
                
                cls, x1, y1, x2, y2 = line.split(" ")
                cls, x1, y1, x2, y2 = int(cls), int(x1), int(y1), int(x2), int(y2)
                if cls in self.annot_cls_maps.keys():
                    boxes[img_file][self.annot_cls_maps[cls]].append([x1, y1, x2, y2])
                
                files.append(img_file)
                    
        return files, boxes
