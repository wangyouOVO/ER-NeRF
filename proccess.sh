CUDA_VISIBLE_DEVICES=1,2,3 python data_utils/process.py data/wenty/wenty.mp4 &&
CUDA_VISIBLE_DEVICES=1,2,3 python main.py data/wenty/ --workspace workspace/wenty/trial_wenty_head -O --iters 100000 &&
CUDA_VISIBLE_DEVICES=1,2,3 python main.py data/wenty/ --workspace workspace/wenty/trial_wenty_head -O --iters 125000 --finetune_lips --patch_size 32
