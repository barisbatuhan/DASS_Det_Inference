import os
from collections import OrderedDict
from loguru import logger

import cv2
import numpy as np

from .datasets_wrapper import Dataset


class ICartoonFaceDataset(Dataset):
    """
    ICartoonFace dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        train=True,
        img_size=(416, 416),
        preproc=None,
        cache=False, # no cache is supported currently
        limit_dataset=None
    ):
        super().__init__(img_size)
        self.class_ids = [0]
        self.class_dict = {"face" : self.class_ids[0]}
        self.img_size = img_size
        self.preproc = preproc   
        self.files, self.annotations = self.load_annotations(data_dir, train)
        self.files = np.asarray(self.files)
        
        if limit_dataset is not None:
            chosen_files = np.random.choice(
                len(self.files), min(limit_dataset, len(self.files)), replace=False)
            self.files = self.files[chosen_files] 
        
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
        anno = np.zeros((len(annots), 5))

        for idx, face in enumerate(annots):
            anno[idx,:4] = face
            anno[idx,4] = self.class_dict["face"]
        
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
    
    
    def load_annotations(self, icf_paths, train :bool):
        # given a path and partition, it loads all the image paths and annots in that partition
        files = []
        boxes = {}
        
        icf_path = icf_paths["icf_train_imgs"] if train else icf_paths["icf_test_imgs"]
        labels_path = icf_paths["icf_train_labels"] if train else icf_paths["icf_test_labels"]
        
        if labels_path is not None:
            
            f = open(labels_path,'r')
            lines = f.readlines()
            f.close()
    
            for line in lines:
                line = line.rstrip().split(',')
                labels = [float(x) for x in line[1:5]]
                person_annot = np.zeros(4)
                person_annot[0:4] = labels[0:4]
                if icf_path + line[0] not in boxes:
                    boxes[icf_path + line[0]] = []
                boxes[icf_path + line[0]].append(person_annot)
            
            for k in boxes.keys():
                boxes[k] = np.array(boxes[k])
            
            files = [*boxes.keys()]
        
        else:
            files = [os.path.join(icf_path, file) for file in os.listdir(icf_path)]
            for file in files:
                boxes[file] = None

        return files, boxes
        