import os
import sys
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import cv2
import data.transforms_bbox as Tr
from data.voc import VOC_box
from configs.defaults import _C
from models.ClsNet_RepVGG_multi_0421 import Labeler, pad_for_grid
from utils.densecrf import DENSE_CRF
from utils.visual import visual_refined_unary, visual_bam, visual_cam
from utils.visual import str2bool, calc_semantic_segmentation_confusion, get_mIOU
import sys

logger = logging.getLogger("stage2")

def main(cfg):
    
    if cfg.SEED:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)

    tr_transforms = Tr.Normalize_Caffe()
    trainset = VOC_box(cfg, tr_transforms)
    train_loader = DataLoader(trainset, batch_size=1)
    Features = [args.Features1, args.Features2, args.Features3, args.Features4]

    ############################deployment#####################
    model = Labeler(cfg.DATA.NUM_CLASSES, None, cfg.MODEL.ROI_SIZE, args.GRID_SIZE, Features).cuda()
    model.eval()
    model.load_state_dict(torch.load(f"./weights/multi/{args.WEIGHTS}"))
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    torch.save(model.state_dict(), f"./weights/multi/{args.NAME}_deploy.pt")
    del model
    # sys.exit(0)
    #####################################################################

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    parser.add_argument("--NAME", type=str, default="train_cam_repvgg_grid_5")
    parser.add_argument("--GRID_SIZE", type=int, default=1, help="select a GPU index")
    parser.add_argument("--WEIGHTS", type=str, default="train_cam_repvgg.pt")
    parser.add_argument("--SCALE", type=float, default=1, help="select a GPU index")
    parser.add_argument("--SAVE_PSEUDO_LABLES", type=str2bool, default=False)
    parser.add_argument("--SAVE_CAM_LABLES", type=str2bool, default=False)
    parser.add_argument("--cam_eval_thresh", type=float, default=0.0)
    parser.add_argument("--EVAL_PSEUDO_LABLES", type=str2bool, default=True)
    parser.add_argument("--EVAL_CAM_LABLES", type=str2bool, default=True)
    parser.add_argument("--EVAL_PATH", type=str, default="/home/ubt/devdata/zdy/BANA_WHU/datasets/WHU/train/label")
    parser.add_argument("--DCRF", nargs="*", type=int, default=[2, 51, 2, 3, 3])
    parser.add_argument("--Features1", type=str2bool, default=False)
    parser.add_argument("--Features2", type=str2bool, default=False)
    parser.add_argument("--Features3", type=str2bool, default=False)
    parser.add_argument("--Features4", type=str2bool, default=False)
    parser.add_argument("--Features5", type=str2bool, default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    main(cfg)