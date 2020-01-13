#!/usr/bin/env sh
python3 main.py --ngpu=4 --test --start_epoch=0 --test_iter=500 --save_iter=1 --workers=4 --batchSize=512 --test_batchSize=256 --nrow=16 --upscale=2 --num_layers_res=2 --input_height=128 --output_height=128 --crop_height=128 --lr=2e-4 --nEpochs=20000 --cuda
