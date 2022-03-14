import os
from collections import OrderedDict
from loguru import logger

import cv2
import numpy as np

import xmltodict

from .datasets_wrapper import Dataset


class Manga109Dataset(Dataset):
    """
    Manga109 dataset class.
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
        self.class_ids = [0, 1, 2]
        self.class_dict = {"frame": self.class_ids[0], 
                           "face" : self.class_ids[1], 
                           "body" : self.class_ids[2]}
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
        # gven a path and partition, it loads all the image paths and annots in that partition
        files = []
        boxes = {}
        
        if type(paths) == str:
            manga109_path = paths
        else:
            manga109_path = paths["m109"]
        
        books = os.listdir(os.path.join(manga109_path, "annotations"))
        books.sort()
        
        test_books = ["UltraEleven", "UnbalanceTokyo", "WarewareHaOniDearu", "YamatoNoHane", "YasasiiAkuma", 
                      "YouchienBoueigumi", "YoumaKourin", "YukiNoFuruMachi", "YumeNoKayoiji", "YumeiroCooking"]
        
        for book in books:
            
            if train and (book[:-4] in test_books):
                continue
            elif (not train) and (book[:-4] not in test_books):
                continue
            
            f = open(os.path.join(manga109_path, "annotations", book), "r")
            book_annots = xmltodict.parse(f.read())
            book_annots = book_annots["book"]["pages"]["page"]
            f.close()
            
            for i, page in enumerate(book_annots):
                
                if i < 10:
                    page_txt = "00" + str(i) + ".jpg"
                elif i < 100:
                    page_txt = "0" + str(i) + ".jpg"
                else:
                    page_txt = str(i) + ".jpg"

                if not train or "face" in page or "body" in page or "frame" in page:
                    k = os.path.join(manga109_path, "images", book[:-4], page_txt)
                    files.append(k)
                    boxes[k] = {"face": [], "body": [], "frame": []}

                if "frame" in page:
                    
                    if type(page["frame"]) == OrderedDict:
                        page["frame"] = [page["frame"]]
                    
                    for el in page["frame"]:
                        xmin = int(el["@xmin"])
                        xmax = int(el["@xmax"])
                        ymin = int(el["@ymin"])
                        ymax = int(el["@ymax"]) 
                        boxes[k]["frame"].append([xmin, ymin, xmax, ymax])
                
                if "face" in page:
                
                    if type(page["face"]) == OrderedDict:
                        page["face"] = [page["face"]]
                    
                    for el in page["face"]:
                        xmin = int(el["@xmin"])
                        xmax = int(el["@xmax"])
                        ymin = int(el["@ymin"])
                        ymax = int(el["@ymax"]) 
                        boxes[k]["face"].append([xmin, ymin, xmax, ymax])
                 
                if "body" in page:
                
                    if type(page["body"]) == OrderedDict:
                        page["body"] = [page["body"]]
                    
                    for el in page["body"]:
                        xmin = int(el["@xmin"])
                        xmax = int(el["@xmax"])
                        ymin = int(el["@ymin"])
                        ymax = int(el["@ymax"])
                        boxes[k]["body"].append([xmin, ymin, xmax, ymax])
                
        return files, boxes
        