# CUDA_VISIBLE_DEVICES=1,2,3 python data_utils/process.py data/wenty/wenty.mp4 &&
# CUDA_VISIBLE_DEVICES=1,2,3 python main.py data/wenty/ --workspace workspace/wenty/trial_wenty_head -O --iters 100000 &&
# CUDA_VISIBLE_DEVICES=1,2,3 python main.py data/wenty/ --workspace workspace/wenty/trial_wenty_head -O --iters 125000 --finetune_lips --patch_size 32
# CUDA_VISIBLE_DEVICES=7 python main.py data/wen1 --use_depth_loss --workspace workspace/wen1/trial_wen1/ -O --iters 100000 &&
# CUDA_VISIBLE_DEVICES=7 python main.py data/wen1 --use_depth_loss --workspace workspace/wen1/trial_wen1/ -O --iters 125000 --finetune_lips --patch_size 32
#训练了身子才需要加 --torso 
# CUDA_VISIBLE_DEVICES=7 python main.py data/wen1/ --workspace workspace/wen1/trial_wen1/ -O --test --test_train --smooth_path --aud ./data/audio/leijun.npy
# ffmpeg -i /dataset/wt/ER-NeRF/workspace/wenty/trial_wenty_depth_3dmmv3/results/ngp_ep0017.mp4 -i /dataset/wt/ER-NeRF/data/audio/leijun.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 wenty_3dnn_v3.mp4
# CUDA_VISIBLE_DEVICES=5 python main.py data/wenty --use_depth_loss --workspace workspace/wenty/trial_wenty_depth_3dmmv3 -O --iters 100000 &&
# CUDA_VISIBLE_DEVICES=5 python main.py data/wenty --use_depth_loss --workspace workspace/wenty/trial_wenty_depth_3dmmv3 -O --iters 125000 --finetune_lips --patch_size 32 &&
# CUDA_VISIBLE_DEVICES=5 python main.py data/wenty --use_depth_loss --workspace workspace/wenty/trial_wenty_depth_3dmmv4 -O --iters 100000 &&
# CUDA_VISIBLE_DEVICES=5 python main.py data/wenty --workspace workspace/wenty/trial_wenty_depth_3dmmv4 -O --iters 125000 --finetune_lips --patch_size 32
# # CUDA_VISIBLE_DEVICES=5 python main.py data/wenty/ --workspace workspace/wenty/trial_wenty_depth_3dmmv3 -O --test --test_train --smooth_path --aud ./data/audio/leijun.npy
# ffmpeg -i /server20/wt/DER-NeRF/output.mp4 -i /dataset/wt/ER-NeRF/wenty_3dnn_v3.mp4 -filter_complex "[0:v]scale=512x512[left]; [1:v]scale=512x512[right]; [left][right]hstack=inputs=2[outv]" -map "[outv]" -c:v libx264 -crf 23 -preset veryfast wenty_comp.mp4
# CUDA_VISIBLE_DEVICES=4 python main.py data/obama --use_depth_loss --workspace workspace/obama/trial_obama_depth_3dmm_backward_d_20 -O --iters 100000 &&
# CUDA_VISIBLE_DEVICES=4 python main.py data/obama --use_depth_loss --workspace workspace/obama/trial_obama_depth_3dmm_backward_d_20 -O --iters 125000 --finetune_lips --patch_size 32 &&
# CUDA_VISIBLE_DEVICES=4 python main.py data/obama --use_depth_loss --workspace workspace/obama/trial_obama_depth_3dmm_backward_nd_20 -O --iters 100000 &&
# CUDA_VISIBLE_DEVICES=4 python main.py data/obama  --workspace workspace/obama/trial_obama_depth_3dmm_backward_nd_20 -O --iters 125000 --finetune_lips --patch_size 32
## CUDA_VISIBLE_DEVICES=7 python main.py data/obama/ --workspace /dataset/wt/DTNerf/workspace/obama/trial_obama_depth_3dmm_backward_index/ -O --infer --test --test_train --smooth_path --aud /dataset/wt/DTNerf/data/noise/audio.npy --aud_index_src /dataset/wt/DTNerf/data/noise/auIndex.npy&&
# ffmpeg -i /dataset/wt/DTNerf/workspace/obama/trial_obama_depth_3dmm_backward_index/results/ngp_ep0018.mp4 -i /dataset/wt/DTNerf/data/noise/audio.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 obama_d_20_train_aud_index.mp4

# ffmpeg -i /server20/wt/DER-NeRF/workspace/obama/trial_obama_head/results/ngp_ep0018.mp4 -i /dataset/wt/ER-NeRF/data/audio/leijun.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 obama_1st.mp4
# ffmpeg -i /dataset/wt/DTNerf/obama_1st.mp4 -i /dataset/wt/DTNerf/workspace/obama/trial_obama_depth_3dmm_backward_d_20/results/ngp_ep0018.mp4 -filter_complex "[0:v]scale=512x512[left]; [1:v]scale=512x512[right]; [left][right]hstack=inputs=2[outv]" -map "[outv]" -c:v libx264 -crf 23 -preset veryfast obama_comp_d_20_v2.mp4 &&
# ffmpeg -i obama_comp_d_20_v2.mp4 -i /dataset/wt/ER-NeRF/data/audio/leijun.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 obama_comp_d_20_train.mp4
CUDA_VISIBLE_DEVICES=7 python main.py data/obama --use_depth_loss --workspace workspace/obama/trial_obama_depth_3dmm_backward_index -O --iters 100000 &&
CUDA_VISIBLE_DEVICES=7 python main.py data/obama --use_depth_loss --workspace workspace/obama/trial_obama_depth_3dmm_backward_index -O --iters 125000 --finetune_lips --patch_size 32