set -exuo pipefail

# export CUDA_VISIBLE_DEVICES=4,5
# train the model
# python train_multi_gpus.py \
#     -i prostate_processed/npy \
#     -task_name MedSAM-Lite-Box \
#     -pretrained_checkpoint lite_medsam.pth \
#     -work_dir ./work_dir_ddp \
#     -batch_size 16 \
#     -num_workers 8 \
#     -lr 0.0005 \
#     --data_aug \
#     -world_size 2 \
#     -node_rank 0 \
#     -init_method tcp://127.0.0.1:29456

# evaluate the model
# remember to change the checkpoint path
python inference_3D.py \
    -data_root prostate_processed/MedSAM_test/MR_Abd \
    -pred_save_dir ./preds/test_npz \
    -medsam_lite_checkpoint_path work_dir_ddp/MedSAM-Lite-Box-20240510-1136/medsam_lite_best.pth \
    -num_workers 4 \
    --save_overlay \
    -png_save_dir ./preds/test_visualization \
    --overwrite
