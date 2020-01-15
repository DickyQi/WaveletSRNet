#!/usr/bin/env sh
python3 main.py --ngpu=4 --test --start_epoch=31 --test_iter=254 --save_iter=1 --workers=4 --batchSize=256 --test_batchSize=64 --nrow=8 --upscale=1 --num_layers_res=1 --input_height=128 --output_height=128 --crop_height=128 --lr=5e-5 --nEpochs=20000 --cuda --pretrained=/home/migu/WaveletSRNet/model/x2/sr_model_epoch_30_iter_0.pth
