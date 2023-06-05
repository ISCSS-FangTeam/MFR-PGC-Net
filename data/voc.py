import os
import collections
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import json
from pycocotools.coco import COCO
import numpy as np
from misc import imutils, edgeutils

class VOC_box(Dataset):
    def __init__(self, cfg, transforms=None):
        if cfg.DATA.MODE == "train":
            json_name = "train.json"
        if cfg.DATA.MODE == "test":
            json_name = "test.json"
        
        f_path = os.path.join(cfg.DATA.ROOT, cfg.DATA.MODE, json_name)
        self.coco = COCO(f_path)
        self.filenames, self.im_dict = self.read_json_filenames(f_path)
        self.transforms = transforms
        
        self.img_path  = os.path.join(cfg.DATA.ROOT, cfg.DATA.MODE, "img/{}")
        self.mask_path = os.path.join(cfg.DATA.ROOT, 'BgMaskfromBoxes_'+ cfg.DATA.MODE + '/{}')
        self.cam_path = os.path.join(cfg.DATA.ROOT, 'grad_cam_repvgg_grid_2_56_2/Y_crf' + '/{}')
        self.len = len(self.filenames)
    
    def __len__(self):
        return self.len
    # 返回img bboxes，bg_mask
    def __getitem__(self, index):
        fn  = self.filenames[index]
        # print(fn, '123')
        # sys.exit(0)
        img = np.array(Image.open(self.img_path.format(fn)), dtype=np.float32) 
        bboxes  = self.load_bboxes(fn)
        bg_mask = np.array(Image.open(self.mask_path.format(fn.split('.')[0] + ".png")), dtype=np.int64)
        #############gcl###########################################
        # _edgemap = np.array(Image.open(self.cam_path.format(fn.split('.')[0] + ".png")).convert('L'), dtype=np.int64)
        # # print(_edgemap.shape)
        # _edgemap = edgeutils.mask_to_onehot(_edgemap, 1)
        # _edgemap = edgeutils.onehot_to_binary_edges(_edgemap, 2, 1)[0]
        # # edgemap1 = Image.fromarray(_edgemap * 255)
        # # edgemap1.save("/home/zdy/dev/BANA_WHU/datasets/" + fn.split('.')[0] + ".png")
        _edgemap = None
        #############gcl############################################
        if self.transforms is not None:
            img, bboxes, bg_mask, edge_mask = self.transforms(img, bboxes, bg_mask, _edgemap)
        return img, bboxes, bg_mask

    # 根据 json_path 加载对应的文件名 和 图片——id字典
    def read_json_filenames(self, json_path):
        filenames = []
        im_dict = {}
        for image_dict in self.coco.dataset['images']:
            filenames.append(image_dict['file_name'])
            im_dict[image_dict['file_name']] = image_dict['id']
        return filenames, im_dict

    # 根据img_name 加载对应的bbox 
    def load_bboxes(self, img_name):
        annIds = self.coco.getAnnIds(imgIds=self.im_dict[img_name], iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        bboxes = []
        for single in anns:
            bb_wmin = float(single['bbox'][0])
            bb_wmax = float(single['bbox'][0] + single['bbox'][2])
            bb_hmin = float(single['bbox'][1])
            bb_hmax = float(single['bbox'][1] + single['bbox'][3])
            cls_num = single['category_id'] # "category_id": 1
            bboxes.append([bb_wmin, bb_hmin, bb_wmax, bb_hmax, cls_num])
        return np.array(bboxes).astype('float32')

class VOC_box_MSF(VOC_box):
    def __init__(self, cfg, transforms=None, scales=(1.0,)):
        self.scales = scales

        super().__init__(cfg, tr_transforms)

    def __getitem__(self, index):
        fn  = self.filenames[index]
        img = np.array(Image.open(self.img_path.format(fn)), dtype=np.float32) 
        bboxes  = self.load_bboxes(fn)
        bg_mask = np.array(Image.open(self.mask_path.format(fn.split('.')[0] + ".png")), dtype=np.int64)
        if self.transforms is not None:
            img, bboxes, bg_mask = self.transforms(img, bboxes, bg_mask)

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.transforms(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        # out = {"name": name_str, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
        #        "label": torch.from_numpy(self.label_list[idx])}

        return ms_img_list, bboxes, bg_mask


class VOC_seg(Dataset):
    def __init__(self, cfg, transforms=None):
        self.train = False
        if cfg.DATA.MODE == "train_weak":
            txt_name = "train_aug.txt"
            self.train = True
        if cfg.DATA.MODE == "val":
            txt_name = "val.txt"
        if cfg.DATA.MODE == "test":
            txt_name = "test.txt"
            
        f_path = os.path.join(cfg.DATA.ROOT, "ImageSets/Segmentation", txt_name)
        self.filenames = [x.split('\n')[0] for x in open(f_path)]
        self.transforms = transforms
        
        self.annot_folders = ["SegmentationClassAug"]
        if cfg.DATA.PSEUDO_LABEL_PATH:
            self.annot_folders = cfg.DATA.PSEUDO_LABEL_PATH
        if cfg.DATA.MODE == "test":
            self.annot_folders = None
        
        self.img_path  = os.path.join(cfg.DATA.ROOT, "JPEGImages", "{}.jpg")
        if self.annot_folder is not None:
            self.mask_paths = [os.path.join(cfg.DATA.ROOT, folder, "{}.png") for folder in self.annot_folders]
        self.len = len(self.filenames)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        fn  = self.filenames[index]
        img = Image.open(self.img_path.format(fn))
        if self.annot_folder is not None:
            masks = [Image.open(mp.format(fn)) for mp in self.mask_paths]
        else:
            masks = None
            
        if self.transforms != None:
            img, masks = self.transforms(img, masks)
        
        return img, masks