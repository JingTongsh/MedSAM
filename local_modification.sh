set -exuo pipefail

# make symbolic link
# ln -s /sharedata/datasets/Renji_MRI/prostate_dataset/all prostate_data

# process data

# python pre_CT_MR.py \
#     -img_path prostate_data/images \
#     -img_name_suffix .mha \
#     -gt_path prostate_data/labels \
#     -gt_name_suffix .mha \
#     -output_path prostate_processed \
#     -num_workers 4 \
#     -modality MR \
#     -anatomy Abd \
#     --save_nii

# process data again

# python npz_to_npy.py \
#     -npz_dir prostate_processed/MedSAM_train/MR_Abd \
#     -npy_dir prostate_processed/npy \
#     -num_workers 4

export CUDA_VISIBLE_DEVICES=2,3
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
    -pred_save_dir ./preds12/test_npz \
    -medsam_lite_checkpoint_path work_dir_lite_medsam/MedSAM-Lite-Box-20240417-1253/medsam_lite_best.pth \
    -num_workers 4 \
    --save_overlay \
    -png_save_dir ./preds12/test_visualization \
    --overwrite
