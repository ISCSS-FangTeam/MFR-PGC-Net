#!/bin/bash

# 多尺度 Features2,3,4 为True
for n in 2
do
	for m in 51
    do
    	for l in 2 
    	do
		python grad_cam_repvgg_multi.py --NAME multi6333_g3_${n}_${m}_${l}  --config-file configs/grad_cam_repvgg.yml \
										--WEIGHTS multi633_g3_deploy.pt --GRID_SIZE 1  --gpu-id 1 \
										--DCRF ${n} ${m} ${l} 3 3 \
										--SAVE_PSEUDO_LABLES True \
										--Features2 True \
										--Features3 True \
										--Features4 True 
		done
	done
done	
