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

    model = Labeler(cfg.DATA.NUM_CLASSES, None, cfg.MODEL.ROI_SIZE, args.GRID_SIZE, Features, deploy=True).cuda()
    
    model.eval()
    model.load_state_dict(torch.load(f"./weights/multi/{args.WEIGHTS}"))

    WEIGHTS_list = []
    

    if args.Features5:
        WEIGHTS_list.append(torch.clone(model.classifier.weight.data).cuda())
    if args.Features1:
        WEIGHTS_1 = torch.clone(model.classifier_1.weight.data).cuda()
        WEIGHTS_list.append(WEIGHTS_1)
    if args.Features2:
        WEIGHTS_2 = torch.clone(model.classifier_2.weight.data).cuda()
        WEIGHTS_list.append(WEIGHTS_2)
    if args.Features3:
        WEIGHTS_3 = torch.clone(model.classifier_3.weight.data).cuda()
        WEIGHTS_list.append(WEIGHTS_3)
    if args.Features4:
        WEIGHTS_4 = torch.clone(model.classifier_4.weight.data).cuda()
        WEIGHTS_list.append(WEIGHTS_4)
    
   
    bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std = args.DCRF
    dCRF = DENSE_CRF(bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)
    
    if args.SAVE_PSEUDO_LABLES:
        folder_name = os.path.join(cfg.DATA.ROOT, args.NAME)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        save_paths1 = []
        for txt in ("Y_crf", ):
            sub_folder = folder_name + f"/{txt}"
            if not os.path.exists(sub_folder):
                os.mkdir(sub_folder)
            save_paths1 += [os.path.join(sub_folder, "{}.png")]

    if args.SAVE_CAM_LABLES:
        folder_name = os.path.join(cfg.DATA.ROOT, args.NAME)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        save_paths2 = []
        for txt in ("CAMs", ):
            sub_folder = folder_name + f"/{txt}"
            if not os.path.exists(sub_folder):
                os.mkdir(sub_folder)
            save_paths2 += [os.path.join(sub_folder, "{}.png")]

    print(f"START {args.NAME} -->")
    preds = []
    preds_cam = []
    gts = []
    with torch.no_grad(): 
        for it, (img, bboxes, bg_mask) in enumerate(tqdm(train_loader)):
            '''
            img     : (1,3,H,W) float32
            bboxes  : (1,K,5)   float32
            bg_mask : (1,H,W)   float32
            '''

            fn = trainset.filenames[it].split(".")[0]
            
            rgb_img = np.array(Image.open(trainset.img_path.format(trainset.filenames[it])))
            bboxes = bboxes[0] # (1,K,5) --> (K,5)
            if bboxes.shape[0] == 0:
                continue
            bg_mask = bg_mask[None] # (1,H,W) --> (1,1,H,W)
            img_H, img_W = img.shape[-2:]
            norm_H, norm_W = (img_H-1)/2, (img_W-1)/2
            bboxes[:,[0,2]] = bboxes[:,[0,2]]*norm_W + norm_W
            bboxes[:,[1,3]] = bboxes[:,[1,3]]*norm_H + norm_H
            bboxes = bboxes.long()
            gt_labels = bboxes[:,4].unique()  
            features_all = model.get_features(img.cuda())
            features_list = []
            if args.Features5:
                features_list.append(features_all[0])
            if args.Features1:
                features_list.append(features_all[1])
            if args.Features2:
                features_list.append(features_all[2])
            if args.Features3:
                features_list.append(features_all[3])
            if args.Features4:
                features_list.append(features_all[4])


            normed_cam = []
            i = 0
            for features, WEIGHTS in zip(features_list, WEIGHTS_list) :
                
                features = F.interpolate(features, img.shape[-2:], mode='bilinear', align_corners=True)
                padded_features = pad_for_grid(features, args.GRID_SIZE)
               
                padded_bg_mask = pad_for_grid(bg_mask.cuda(), args.GRID_SIZE)
                
                grid_bg, valid_gridIDs = model.get_grid_bg_and_IDs(padded_bg_mask, args.GRID_SIZE)
                
                bg_protos = model.get_bg_prototypes(padded_features, padded_bg_mask, grid_bg, args.GRID_SIZE)
                 
                bg_protos = bg_protos[0,valid_gridIDs] 
                  
                normed_bg_p = F.normalize(bg_protos)
                normed_f = F.normalize(features)
                
                
                bg_attns = F.relu(torch.sum(normed_bg_p*normed_f, dim=1))
                bg_attn = torch.mean(bg_attns, dim=0, keepdim=True) 
                bg_attn[bg_attn < cfg.MODEL.BG_THRESHOLD * bg_attn.max()] = 0
                
                Bg_unary = torch.clone(bg_mask[0]) 
                
                region_inside_bboxes = Bg_unary[0]==0 
                Bg_unary[:,region_inside_bboxes] = bg_attn[:,region_inside_bboxes].detach().cpu()

                Fg_unary = []
                w_c = WEIGHTS[1][None]
                raw_cam= F.relu(torch.sum(w_c*features, dim=1))
                i += 1

                normed_cam_single = torch.zeros_like(raw_cam)
                for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==1]:
                    denom = raw_cam[:,hmin:hmax,wmin:wmax].max() + 1e-12
                    normed_cam_single[:,hmin:hmax,wmin:wmax] = raw_cam[:,hmin:hmax,wmin:wmax] / denom
                normed_cam.append(normed_cam_single)
            
            normed_cam_sum = normed_cam[0] + normed_cam[1] + normed_cam[2]
            
            cam = torch.zeros_like(normed_cam_sum)
            for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==1]:
                denom = normed_cam_sum[:,hmin:hmax,wmin:wmax].max() + 1e-12
                cam[:,hmin:hmax,wmin:wmax] = normed_cam_sum[:,hmin:hmax,wmin:wmax] / denom

            ########################################################
            if args.EVAL_CAM_LABLES:
                cam = (normed_cam_sum).detach().cpu().numpy()
                cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thresh)
                cls_labels = np.argmax(cam, axis=0)
                keys = np.array([0, 1])
                cls_labels = keys[cls_labels]
                if args.SAVE_CAM_LABLES:
                    for pseudo, save_path in zip([cls_labels], save_paths2):
                        # pseudo *= 255
                        Image.fromarray(np.uint8(pseudo)*255).save(save_path.format(fn))
                        # sys.exit(0)
                preds_cam.append(cls_labels)
                gt = cv2.imread(f'{args.EVAL_PATH}/{fn}.tif', cv2.IMREAD_GRAYSCALE) // 255
                gts.append(gt)
                continue
            #########################################################
            
            Fg_unary += [cam]
            Fg_unary = torch.cat(Fg_unary, dim=0).detach().cpu()
            unary = torch.cat((Bg_unary,Fg_unary), dim=0)
            unary[:,region_inside_bboxes] = torch.softmax(unary[:,region_inside_bboxes], dim=0)

            
            refined_unary = dCRF.inference(rgb_img, unary.numpy())

            
            for idx_cls, uni_cls in enumerate(gt_labels,1):
                mask = np.zeros((img_H,img_W))
                for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                    mask[hmin:hmax,wmin:wmax] = 1
                refined_unary[idx_cls] *= mask    
            
            tmp_mask = refined_unary.argmax(0)
            Y_crf = np.zeros_like(tmp_mask, dtype=np.uint8)
            for idx_cls, uni_cls in enumerate(gt_labels,1):
                Y_crf[tmp_mask==idx_cls] = uni_cls
            Y_crf[tmp_mask==0] = 0

            ######################correction###########################################
            cam = np.pad(normed_cam_sum.detach().cpu(), ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thresh)
            cls_labels = np.argmax(cam, axis=0)
            keys = np.array([0, 1])
            cls_labels = keys[cls_labels]

            for idx_cls, uni_cls in enumerate(gt_labels,1):
                for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                    # area = (hmax-hmin)*(wmax-wmin)
                    if np.sum(Y_crf[hmin:hmax,wmin:wmax]) == 0:
                        Y_crf[hmin:hmax,wmin:wmax] = cls_labels[hmin:hmax,wmin:wmax]
            ##############################################################################
            
            if args.SAVE_PSEUDO_LABLES:
                for pseudo, save_path in zip([Y_crf], save_paths1):
                    pseudo *= 1
                    Image.fromarray(pseudo).save(save_path.format(fn))
                    
            #########################################################
            if args.EVAL_PSEUDO_LABLES:
                preds.append(Y_crf)
                gt = cv2.imread(f'{args.EVAL_PATH}/{fn}.tif', cv2.IMREAD_GRAYSCALE) // 255
                gts.append(gt)
            #########################################################
            
        get_mIOU(preds, gts, args.NAME)
    logger.info(f"END {args.NAME} -->")


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