import os
import collections
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset


CLASSES = (
    "background", 
    "aeroplane", 
    "bicycle", 
    "bird", 
    "boat", 
    "bottle", 
    "bus", 
    "car", 
    "cat", 
    "chair", 
    "cow", 
    "diningtable", 
    "dog", 
    "horse", 
    "motorbike",
    "person",
    "pottedplant", 
    "sheep",
    "sofa", 
    "train",
    "tvmonitor"
)

palette = [0,0,0, 255,255,255]


def parse_voc_xml(node):
    voc_dict = {}
    children = list(node)
    if children:
        def_dic = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        voc_dict = {
            node.tag:
                {ind: v[0] if len(v) == 1 else v
                 for ind, v in def_dic.items()}
        }
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict

def load_bboxes(self, img_name):
    # 根据图片名称找到该张图片的bbox
    with open(json_path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        ann_list = json_data['annotations']
        bboxes = []
        for ind, image_dict in enumerate(ann_list):
            if image_dict['image_id'] == self.im_dict[img_name]:
                bbox = image_dict['bbox']  # xywh
                bb_wmin = float(image_dict['bbox'][0])
                bb_wmax = float(image_dict['bbox'][0] + image_dict['bbox'][2])
                bb_hmin = float(image_dict['bbox'][1])
                bb_hmax = float(image_dict['bbox'][1] + image_dict['bbox'][3])
                cls_num = image_dict['category_id'] # "category_id": 1
                bboxes.append([bb_wmin, bb_hmin, bb_wmax, bb_hmax, cls_num])
        return np.array(bboxes).astype('float32')

def get_single_binaryImg(json_path, binary_img_save):
    
    dir = os.listdir(json_path)  
    for jfile in dir:
        annFile = os.path.join(json_path, jfile)
        print(annFile)
        coco = COCO(annFile)
        imgIds = coco.getImgIds()  
        img = coco.loadImgs(imgIds[0])[0]
        
        catIds = []
        for ann in coco.dataset['annotations']:
            if ann['image_id'] == imgIds[0]:
                catIds.append(ann['category_id'])

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        width = img['width']
        height = img['height']
        anns = coco.loadAnns(annIds)  
        mask_pic = np.zeros((height, width))
        for single in anns: 
            mask_single = coco.annToMask(single)
            mask_pic += mask_single

        for row in range(height):
            for col in range(width):
                if (mask_pic[row][col] > 0):
                    mask_pic[row][col] = 1

        imgs = np.zeros(shape=(height, width, 3), dtype=np.float32)
        imgs[:, :, 0] = mask_pic[:, :]
        imgs[:, :, 1] = mask_pic[:, :]
        imgs[:, :, 2] = mask_pic[:, :]
        # imgs = imgs.astype(int)
        img_name = img['file_name'].split(".")[0]
        try:
            plt.imsave(binary_img_save + img_name + ".png", imgs)
        except:
            print('')
        # break

if __name__ == '__main__':
    # trainval.txt 文件是voc所有的文件
    json_path = "/home/ubt/devdata/zdy/BANA_WHU/datasets/WHU/train.json"
    save_path = os.path.join('/home/ubt/devdata/VOCdevkit/VOC2012', 'BgMaskfromBoxes/{}.png')
    get_single_binaryImg(json_path, save_path)
    filenames = [x.split("\n")[0] for x in open(f_path)]
    xml_path  = os.path.join('/home/ubt/devdata/VOCdevkit/VOC2012', 'Annotations/{}.xml')
    for ind, file in enumerate(filenames):
        w, h, bboxes = load_bboxes(xml_path.format(file))
        Bgmask = np.ones((int(h), int(w)))
        for box in bboxes:
            Bgmask[box[1]-1:box[3], box[0]-1:box[2]] = 0.0 # 这里注意一下坐标
        # out = Image.fromarray(Bgmask.astype(np.uint8))
        # out.save(save_path.format(file))

        out_p = Image.fromarray(Bgmask.astype(np.uint8), mode='P')
        out_p.putpalette(palette)
        out_p.save(save_path.format(file))
        print('{}success'.format(file), ind)
        # sys.exit(0)
