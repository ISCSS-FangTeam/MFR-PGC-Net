import torch
import torch.nn as nn
import torch.nn.functional as F

from .RepVGG import repvgg_model_convert, create_RepVGG_B1g2, create_RepVGG_B2

class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """
#     BATCH_NORM = _BATCH_NORM
    def __init__(
        self, in_ch, norm_type, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", norm_type(out_ch, eps=1e-5, momentum=0.999))
        if relu:
            self.add_module("relu", nn.ReLU())

def pad_for_grid(mask, grid_size):
    Pad_H = grid_size - mask.shape[2] % grid_size  # 求余 1
    Pad_W = grid_size - mask.shape[3] % grid_size  # 求余 1
    if Pad_H == grid_size:
        Pad_H = 0
    if Pad_W == grid_size:
        Pad_W = 0
    if Pad_H % 2 == 0:
        if Pad_W % 2 == 0:
            out = F.pad(mask, [Pad_W // 2, Pad_W // 2, Pad_H // 2, Pad_H // 2], value=0)
        else:
            out = F.pad(mask, [0, Pad_W, Pad_H // 2, Pad_H // 2], value=0)
    else:
        if Pad_W % 2 == 0:
            out = F.pad(mask, [Pad_W // 2, Pad_W // 2, 0, Pad_H], value=0)
        else:
            out = F.pad(mask, [0, Pad_W, 0, Pad_H], value=0)
    return out


class Labeler(nn.Module):
    def __init__(self, num_classes, backbone_file, roi_size, grid_size, Features, deploy=False, pretrained=False):
        # 21 [2,2] 4
        super().__init__()
        # self.backbone = VGG16(dilation=1)  # 1024
        backbone = create_RepVGG_B1g2(deploy)
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
        self.classifier_1 = nn.Conv2d(64, num_classes, 1, bias=False)
        self.classifier_2 = nn.Conv2d(128, num_classes, 1, bias=False)
        self.classifier_3 = nn.Conv2d(256, num_classes, 1, bias=False)
        self.classifier_4 = nn.Conv2d(512, num_classes, 1, bias=False)

        self.OH, self.OW = roi_size  # 2 2
        self.GS = grid_size
        self.GS_1 = grid_size
        self.GS_2 = grid_size * 2
        self.GS_3 = grid_size 
        self.GS_4 = grid_size
        self.from_scratch_layers = [self.classifier, self.classifier_1, self.classifier_2, self.classifier_3, self.classifier_4]
        # self.newly_added = nn.ModuleList([self.classifier])

        self.Features = Features
        # print(self.Features)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)
        # backbone.load_state_dict(torch.load(backbone_file))  

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4

        for n, m in self.layer3.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                # print(m.dilation, m.padding, m.stride, m.kernel_size)  # (1, 1) (1, 1) (2, 2)
                m.dilation, m.padding, m.stride = (1, 1), (1, 1), (1, 1)
                print('change dilation, padding, stride of ', n)
            elif 'rbr_1x1' in n and isinstance(m, nn.Conv2d):
                m.stride = (1, 1)
                print('change stride of ', n)
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                m.dilation, m.padding, m.stride = (1, 1), (1, 1), (1, 1)
                # m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                print('change dilation, padding, stride of ', n)
            elif 'rbr_1x1' in n and isinstance(m, nn.Conv2d):
                m.stride = (1, 1)
                print('change stride of ', n)              

    def get_features(self, x):

        x1 = self.layer0(x)  
        # print(x.shape)
        x2 = self.layer1(x1)  
        
        # print(x2.shape)
        x3 = self.layer2(x2)  
        # print(x3.shape)
        x4 = self.layer3(x3)  
        # print(x.shape)
        x = self.layer4(x4)      
        return x, x1, x2, x3, x4

    def weighted_avg_pool_2d(self, input, weight):
        filtered = input * weight
        _, dims, input_H, input_W = filtered.shape
        # print(filtered.shape)  # torch.Size([1, 1024, 10, 13])
        stride_H = input_H // self.OH  
        stride_W = input_W // self.OW
        # print(stride_H, stride_W)  # 5 6
        if stride_H == 0:
            stride_H += 1
            pad_H = self.OH - input_H
            filtered = F.pad(filtered, [0, 0, 0, pad_H], mode='replicate')
            weight = F.pad(weight, [0, 0, 0, pad_H], mode='replicate')
        if stride_W == 0:
            stride_W += 1
            pad_W = self.OW - input_W
            filtered = F.pad(filtered, [pad_W, 0, 0, 0], mode='replicate')
            weight = F.pad(weight, [pad_W, 0, 0, 0], mode='replicate')
        ks_H = input_H - (self.OH - 1) * stride_H
        ks_W = input_W - (self.OW - 1) * stride_W
        # print(ks_H, ks_W)  # 5, 7
        if ks_H <= 0:
            ks_H = 1
        if ks_W <= 0:
            ks_W = 1
        kernel = torch.ones((dims, 1, ks_H, ks_W)).type_as(filtered)  # torch.Size([1024, 1, 5, 7])
        
        numer = F.conv2d(filtered, kernel, stride=(stride_H, stride_W), groups=dims)
        denom = F.conv2d(weight, kernel[0][None], stride=(stride_H, stride_W)) + 1e-12
        return numer / denom

    def gen_grid(self, box_coord, width, height):
        wmin, hmin, wmax, hmax = box_coord[:4]
        # print(box_coord[:4])  
        grid_x = torch.linspace(wmin, wmax, width).view(1, 1, width, 1)
        # print(grid_x.shape)  
        grid_y = torch.linspace(hmin, hmax, height).view(1, height, 1, 1)
        grid_x = grid_x.expand(1, height, width, 1)
        # expand 复制以扩充维度
        grid_y = grid_y.expand(1, height, width, 1)
        grid = torch.cat((grid_x, grid_y), dim=-1)  
        return grid

    def BAP(self, features, bboxes, batchID_of_box, bg_protos, valid_cellIDs, ind_valid_bg_mask):
        batch_size, _, fH, fW = features.shape
        norm_H, norm_W = (fH - 1) / 2, (fW - 1) / 2  # 20
        widths = bboxes[:, [0, 2]] * norm_W + norm_W
        heights = bboxes[:, [1, 3]] * norm_H + norm_H
        widths = (widths[:, 1].ceil() - widths[:, 0].floor()).int()  
        heights = (heights[:, 1].ceil() - heights[:, 0].floor()).int()
        fg_protos = []
        for batch_id in range(batch_size):
            feature_map = features[batch_id][None] 
            indices = batchID_of_box == batch_id  
            for coord, width, height in zip(bboxes[indices], widths[indices], heights[indices]):
                
                grid = self.gen_grid(coord, width, height).type_as(feature_map)
                roi = F.grid_sample(feature_map, grid) 
                GAP_attn = torch.ones(1, 1, *roi.shape[-2:]).type_as(roi)
                ID_list = valid_cellIDs[batch_id]
                if ind_valid_bg_mask[batch_id] and len(ID_list):
                    normed_roi = F.normalize(roi, dim=1)  
                    valid_bg_p = bg_protos[batch_id, ID_list]  
                    normed_bg_p = F.normalize(valid_bg_p, dim=1)
                    bg_attns = F.relu(torch.sum(normed_roi * normed_bg_p, dim=1, keepdim=True))
                
                    bg_attn = torch.mean(bg_attns, dim=0, keepdim=True)
                    fg_attn = 1 - bg_attn
                    fg_by_BAP = self.weighted_avg_pool_2d(roi, fg_attn)  # (1,256,OH,OW)
                    fg_protos.append(fg_by_BAP)
                else:
                    fg_by_GAP = self.weighted_avg_pool_2d(roi, GAP_attn)  # (1,256,OH,OW)
                    fg_protos.append(fg_by_GAP)
        fg_protos = torch.cat(fg_protos, dim=0)
        return fg_protos

    def get_grid_bg_and_IDs(self, padded_mask, grid_size):
        batch_size, _, padded_H, padded_W = padded_mask.shape
        cell_H, cell_W = padded_H // grid_size, padded_W // grid_size  
        grid_bg = padded_mask.unfold(2, cell_H, cell_H).unfold(3, cell_W, cell_W)
        grid_bg = torch.sum(grid_bg, dim=(4, 5))  
        grid_bg = grid_bg.view(-1, 1, 1, 1)  # (N * GS**2,1,1,1)
        valid_gridIDs = [idx for idx, cell in enumerate(grid_bg) if cell > 0]
        grid_bg = grid_bg.view(batch_size, -1, 1, 1, 1)  # (N,GS**2,1,1,1)
        return grid_bg, valid_gridIDs

    def get_bg_prototypes(self, padded_features, padded_mask, denom_grids, grid_size):
        batch_size, dims, padded_H, padded_W = padded_features.shape
        cell_H, cell_W = padded_H // grid_size, padded_W // grid_size
        bg_features = (padded_mask * padded_features).unfold(2, cell_H, cell_H).unfold(3, cell_W, cell_W)
        bg_protos = torch.sum(bg_features, dim=(4, 5))  # (N,dims,GS,GS,cH,cW) --> (N,dims,GS,GS)
        bg_protos = bg_protos.view(batch_size, dims, -1).permute(0, 2, 1)  # (N,GS*GS,dims)
        bg_protos = bg_protos.contiguous().view(batch_size, -1, dims, 1, 1)
        bg_protos = bg_protos / (denom_grids + 1e-12)  # (N,GS**2,dims,1,1)  # denom_grids就是grid_bg
        return bg_protos

    def forward(self, img, bboxes, batchID_of_box, bg_mask, ind_valid_bg_mask):
        '''
        img               : (N,3,H,W) float32
        bboxes            : (K,5) float32
        batchID_of_box    : (K,) int64
        bg_mask           : (N,1,H,W) float32
        ind_valid_bg_mask : (N,) uint8
        '''
        features, features_1, features_2, features_3, features_4 = self.get_features(img) 
        out, out_1, out_2, out_3, out_4 = None, None, None, None, None
        if self.Features[4]:
            batch_size, dims, fH, fW = features.shape
            ##########################################################
            padded_mask = pad_for_grid(F.interpolate(bg_mask, (fH, fW)), self.GS)
            # print(padded_mask.shape)  
            grid_bg, valid_gridIDs = self.get_grid_bg_and_IDs(padded_mask, self.GS)
            valid_cellIDs = []
            for grids in grid_bg:  # # (N,GS**2,1,1,1)
                valid_cellIDs.append([idx for idx, cell in enumerate(grids) if cell > 0])
            ##########################################################
            padded_features = pad_for_grid(features, self.GS)
            bg_protos = self.get_bg_prototypes(padded_features, padded_mask, grid_bg, self.GS)
            fg_protos = self.BAP(features, bboxes, batchID_of_box, bg_protos, valid_cellIDs, ind_valid_bg_mask)
            ##########################################################
            num_fgs = fg_protos.shape[0]
            fg_protos = fg_protos.view(num_fgs, dims, -1).permute(0, 2, 1).contiguous().view(-1, dims, 1, 1)
            bg_protos = bg_protos.contiguous().view(-1, dims, 1, 1)[valid_gridIDs]  
            protos = torch.cat((fg_protos, bg_protos), dim=0)
            out = self.classifier(protos)    

        ###############################################
        if self.Features[0]:
            batch_size, dims, fH, fW = features_1.shape
            padded_mask_1 = pad_for_grid(F.interpolate(bg_mask, (fH, fW)), self.GS_1)
            grid_bg_1, valid_gridIDs_1 = self.get_grid_bg_and_IDs(padded_mask_1, self.GS_1)
            valid_cellIDs_1 = []
            for grids in grid_bg_1:  # # (N,GS**2,1,1,1)
                valid_cellIDs_1.append([idx for idx, cell in enumerate(grids) if cell > 0])

            padded_features_1 = pad_for_grid(features_1, self.GS_1)
            bg_protos_1 = self.get_bg_prototypes(padded_features_1, padded_mask_1, grid_bg_1, self.GS_1)
            fg_protos_1 = self.BAP(features_1, bboxes, batchID_of_box, bg_protos_1, valid_cellIDs_1, ind_valid_bg_mask)
            num_fgs = fg_protos_1.shape[0]
            fg_protos_1 = fg_protos_1.view(num_fgs, dims, -1).permute(0, 2, 1).contiguous().view(-1, dims, 1, 1)

            bg_protos_1 = bg_protos_1.contiguous().view(-1, dims, 1, 1)[valid_gridIDs_1]  # (158,1024,1,1)
            protos_1 = torch.cat((fg_protos_1, bg_protos_1), dim=0)
            out_1 = self.classifier_1(protos_1)
        ###############################################


        ###############################################
        if self.Features[1]:
            batch_size, dims, fH, fW = features_2.shape
            padded_mask_2 = pad_for_grid(F.interpolate(bg_mask, (fH, fW)), self.GS_2)
            grid_bg_2, valid_gridIDs_2 = self.get_grid_bg_and_IDs(padded_mask_2, self.GS_2)
            valid_cellIDs_2 = []
            for grids in grid_bg_2:  # (N,GS**2,1,1,1)
                valid_cellIDs_2.append([idx for idx, cell in enumerate(grids) if cell > 0])

            padded_features_2 = pad_for_grid(features_2, self.GS_2)
            bg_protos_2 = self.get_bg_prototypes(padded_features_2, padded_mask_2, grid_bg_2, self.GS_2)
            fg_protos_2 = self.BAP(features_2, bboxes, batchID_of_box, bg_protos_2, valid_cellIDs_2, ind_valid_bg_mask)
            num_fgs = fg_protos_2.shape[0]
            fg_protos_2 = fg_protos_2.view(num_fgs, dims, -1).permute(0, 2, 1).contiguous().view(-1, dims, 1, 1)

            bg_protos_2 = bg_protos_2.contiguous().view(-1, dims, 1, 1)[valid_gridIDs_2]  
            protos_2 = torch.cat((fg_protos_2, bg_protos_2), dim=0)
            out_2 = self.classifier_2(protos_2)
        ###############################################

        ###############################################
        if self.Features[2]:
            
            batch_size, dims, fH, fW = features_3.shape
            padded_mask_3 = pad_for_grid(F.interpolate(bg_mask, (fH, fW)), self.GS_3)
            grid_bg_3, valid_gridIDs_3 = self.get_grid_bg_and_IDs(padded_mask_3, self.GS_3)
            valid_cellIDs_3 = []
            for grids in grid_bg_3:  # # (N,GS**2,1,1,1)
                valid_cellIDs_3.append([idx for idx, cell in enumerate(grids) if cell > 0])

            padded_features_3 = pad_for_grid(features_3, self.GS_3)
            bg_protos_3 = self.get_bg_prototypes(padded_features_3, padded_mask_3, grid_bg_3, self.GS_3)
            fg_protos_3 = self.BAP(features_3, bboxes, batchID_of_box, bg_protos_3, valid_cellIDs_3, ind_valid_bg_mask)
            num_fgs = fg_protos_3.shape[0]
            fg_protos_3 = fg_protos_3.view(num_fgs, dims, -1).permute(0, 2, 1).contiguous().view(-1, dims, 1, 1)

            bg_protos_3 = bg_protos_3.contiguous().view(-1, dims, 1, 1)[valid_gridIDs_3]  # (158,1024,1,1)
            protos_3 = torch.cat((fg_protos_3, bg_protos_3), dim=0)
            out_3 = self.classifier_3(protos_3)
        ###############################################

        ###############################################
        if self.Features[3]:
            batch_size, dims, fH, fW = features_4.shape
            padded_mask_4 = pad_for_grid(F.interpolate(bg_mask, (fH, fW)), self.GS_4)
            grid_bg_4, valid_gridIDs_4 = self.get_grid_bg_and_IDs(padded_mask_4, self.GS_4)
            valid_cellIDs_4 = []
            for grids in grid_bg_4:  # # (N,GS**2,1,1,1)
                valid_cellIDs_4.append([idx for idx, cell in enumerate(grids) if cell > 0])

            padded_features_4 = pad_for_grid(features_4, self.GS_4)
            bg_protos_4 = self.get_bg_prototypes(padded_features_4, padded_mask_4, grid_bg_4, self.GS_4)
            fg_protos_4 = self.BAP(features_4, bboxes, batchID_of_box, bg_protos_4, valid_cellIDs_4, ind_valid_bg_mask)
            num_fgs = fg_protos_4.shape[0]
            fg_protos_4 = fg_protos_4.view(num_fgs, dims, -1).permute(0, 2, 1).contiguous().view(-1, dims, 1, 1)

            bg_protos_4 = bg_protos_4.contiguous().view(-1, dims, 1, 1)[valid_gridIDs_4]  # (158,1024,1,1)
            protos_4 = torch.cat((fg_protos_4, bg_protos_4), dim=0)
            out_4 = self.classifier_4(protos_4)
        ###############################################

        return [out, out_1, out_2, out_3, out_4]

    def get_params(self, do_init=True):
        '''
        This function is borrowed from AffinitNet. It returns (pret_weight, pret_bias, scratch_weight, scratch_bias).
        Please, also see the paper (Learning Pixel-level Semantic Affinity with Image-level Supervision, CVPR 2018), and codes (https://github.com/jiwoon-ahn/psa/tree/master/network).
        '''
        params = ([], [], [], [])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m in self.from_scratch_layers:
                    if do_init:
                        nn.init.normal_(m.weight, std=0.01)
                    params[2].append(m.weight)
                else:
                    params[0].append(m.weight)
                if m.bias is not None:
                    if m in self.from_scratch_layers:
                        if do_init:
                            nn.init.constant_(m.bias, 0)
                        params[3].append(m.bias)
                    else:
                        params[1].append(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                if m in self.from_scratch_layers:
                    if do_init:
                        nn.init.constant_(m.weight, 1)
                    params[2].append(m.weight)
                else:
                    params[0].append(m.weight)
                if m.bias is not None:
                    if m in self.from_scratch_layers:
                        if do_init:
                            nn.init.constant_(m.bias, 0)
                        params[3].append(m.bias)
                    else:
                        params[1].append(m.bias)
        return params
