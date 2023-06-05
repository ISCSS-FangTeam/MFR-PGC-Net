#!/bin/bash

# 训练多尺度命令
for grid in 3
do
	# layer3为2 layer4为1
	python train_cam_multi.py --NAME multi633_g${grid} \
							  --config-file configs/train_cam_repvgg_321.yml --GRID_SIZE ${grid} \
							  --gpu-id 1 --ROI_SIZE 22 --Features2 True --Features3 True --Features4 True
done

