import os
from collections import OrderedDict
from loguru import logger

import cv2
import numpy as np

from .datasets_wrapper import Dataset


class Comic2kDataset(Dataset):
    """
    Comic2k dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        train=True,
        img_size=(416, 416),
        preproc=None,
        cache=False, # no cache is supported currently
        filter=None, # to filter a subset: "comic", "watercolor", "clipart"
        limit_dataset :int=None
    ):
        super().__init__(img_size)
        self.class_ids = [0]
        self.class_dict = {"body" : self.class_ids[0]}
        self.img_size = img_size
        self.preproc = preproc   
        self.files, self.annotations = self.load_annotations(data_dir, train, filter)
        self.files = np.asarray(self.files)
        
        if limit_dataset is not None:
            self.files = self.files[np.random.choice(len(self.files), 
                                                     min(limit_dataset, len(self.files)), 
                                                     replace=False)]
        
        
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
        domain, root, file = self.files[index]
        annots = self.annotations[domain][file]
        anno = np.zeros((len(annots), 5))

        for idx, body in enumerate(annots):
            anno[idx,:4] = body
            anno[idx,4] = self.class_dict["body"]
        
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
        _, root, file = self.files[index]
        img_path = os.path.join(root, file + ".jpg")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert img is not None
        h, w, c = img.shape
        return img, [h, w]
    
    
    def load_annotations(self, comic2k_paths, train :bool, filter):
        # gven a path and partition, it loads all the image paths and annots in that partition
        files = []
        boxes = {}
        
        annot_final = "train.txt" if train else "test.txt"
        
        # Comic Part Reading
        if filter is None or filter == "comic":
            f = open(os.path.join(comic2k_paths["comic"], "ImageSets/Main", "annotated_" +  annot_final), "r")
            comic2k_files = f.readlines()
            f.close()
            boxes["comic"] = self.__read_xmls__(os.path.join(comic2k_paths["comic"], "Annotations"), comic2k_files, train)
            
            for file in boxes["comic"].keys():
                files.append(["comic", os.path.join(comic2k_paths["comic"], "JPEGImages"), file])
        
        # Watercolor Part Reading
        if filter is None or filter == "watercolor":
            f = open(os.path.join(comic2k_paths["watercolor"], "ImageSets/Main", "annotated_" +  annot_final), "r")
            comic2k_files = f.readlines()
            f.close()
            boxes["watercolor"] = self.__read_xmls__(os.path.join(comic2k_paths["watercolor"], "Annotations"), comic2k_files, train)
            
            for file in boxes["watercolor"].keys():
                files.append(["watercolor", os.path.join(comic2k_paths["watercolor"], "JPEGImages"), file])
        
        
        # Clipart Part Reading
        if filter is None or filter == "clipart":
            f = open(os.path.join(comic2k_paths["clipart"], "ImageSets/Main", annot_final), "r")
            comic2k_files = f.readlines()
            f.close()
            boxes["clipart"] = self.__read_xmls__(os.path.join(comic2k_paths["clipart"], "Annotations"), comic2k_files, train)
            
            for file in boxes["clipart"].keys():
                files.append(["clipart", os.path.join(comic2k_paths["clipart"], "JPEGImages"), file])

        return files, boxes
    
    
    def __read_xmls__(self, annot_path, files, train :bool):
        
        # allowed_cls = ["person", "bird", "cat", "cow", "dog", "horse", "sheep"]
        allowed_cls = ["person"]
        
        boxes = {}
        
        for annot in files:
            anno = annot[:-1]
            boxes[anno] = []
            past_name = None
            new_box = [0, 0, 0, 0]
            
            f = open(os.path.join(annot_path, anno + ".xml"), "r")
            lines = f.readlines()
            f.close()
            for line in lines:
                start = line.find(">") + 1
                end = line.find("</")
                content = line[start:end]
                
                
                if "<name>" in line:
                    if past_name in allowed_cls:
                        if (new_box[3] - new_box[1]) * (new_box[2] - new_box[0]) > 16:
                            # eliminate really small people annotations
                            boxes[anno].append(new_box)
                    
                    new_box = [0, 0, 0, 0]
                    past_name = content
                
                elif past_name in allowed_cls:
                    if "xmin" in line:
                        new_box[0] = int(content)
                    elif "ymin" in line:
                        new_box[1] = int(content)
                    elif "xmax" in line:
                        new_box[2] = int(content)
                    elif "ymax" in line:
                        new_box[3] = int(content)
                
                
            if new_box[-1] != 0 and (new_box[3] - new_box[1]) * (new_box[2] - new_box[0]) > 16:
                boxes[anno].append(new_box)
        
        
        if train:
            all_files = list(boxes.keys())
            for anno in all_files:
                 if len(boxes[anno]) == 0:
                    boxes.pop(anno)
            
        return boxes
        