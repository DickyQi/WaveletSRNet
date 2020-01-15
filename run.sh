#!/usr/bin/env sh
python3 main.py --ngpu=4 --test --start_epoch=186 --test_iter=127 --save_iter=1 --workers=16 --batchSize=512 --test_batchSize=256 --nrow=16 --upscale=2 --num_layers_res=2 --input_height=128 --output_height=128 --crop_height=128 --lr=5e-6 --nEpochs=20000 --cuda --pretrained=/home/migu/WaveletSRNet/model/x4/sr_model_epoch_185_iter_0.pth
