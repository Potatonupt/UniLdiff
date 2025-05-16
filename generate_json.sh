#!/bin/bash

# 每个任务指定不同的 GPU
CUDA_VISIBLE_DEVICES=0 python test_generate_caption.py \
    --img_dir='/data/czh/data/train/allinone/low-light/GT' \
    --save_dir=./json/low-light &

CUDA_VISIBLE_DEVICES=1 python test_generate_caption.py \
    --img_dir='/data/czh/data/train/allinone/motion-blurry/GT' \
    --save_dir=./json/motion-blurry &

CUDA_VISIBLE_DEVICES=2 python test_generate_caption.py \
    --img_dir='/data/czh/data/train/allinone/OTS/GT' \
    --save_dir=./json/OTS &

CUDA_VISIBLE_DEVICES=3 python test_generate_caption.py \
    --img_dir='/data/czh/data/train/allinone/BSDWED15/GT' \
    --save_dir=./json/noise &

CUDA_VISIBLE_DEVICES=4 python test_generate_caption.py \
    --img_dir='/data/czh/data/train/allinone/Rain100L/GT' \
    --save_dir=./json/rain100l &

CUDA_VISIBLE_DEVICES=4 python test_generate_caption.py \
    --img_dir='/data/czh/data/test/rainy/GT' \
    --save_dir=./json_test/rain100l &

CUDA_VISIBLE_DEVICES=3 python test_generate_caption.py \
    --img_dir='/data/czh/data/test/motion-blurry/GT' \
    --save_dir=./json_test/motion-blurry &

CUDA_VISIBLE_DEVICES=0 python test_generate_caption.py \
    --img_dir='/data/czh/data/test/low-light/GT' \
    --save_dir=./json_test/low-light &

CUDA_VISIBLE_DEVICES=1 python test_generate_caption.py \
    --img_dir='/data/czh/data/test/noisy50/GT' \
    --save_dir=./json_test/noise &

# 等待所有后台任务完成
wait
echo "✅ 所有任务完成！"
