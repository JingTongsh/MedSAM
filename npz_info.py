import numpy as np
import os
import glob


def main():
    dir_pred = 'preds13/test_npz/MR_Abd'
    dir_gt = 'prostate_foreground/MedSAM_test/MR_Abd'
    npz_pred_list = glob.glob(os.path.join(dir_pred, '*.npz'))
    npz_gt_list = glob.glob(os.path.join(dir_gt, '*.npz'))
    npz_pred_list.sort()
    npz_gt_list.sort()
    
    for pred_file, gt_file in zip(npz_pred_list, npz_gt_list):
        npz_pred = np.load(pred_file)
        npz_gt = np.load(gt_file)
        pred_arr = npz_pred['segs']
        gt_arr = npz_gt['gts']
        if pred_arr.shape != gt_arr.shape:
            print(f"Shape mismatch: {pred_file}, {gt_file}")
            print(pred_arr.shape, gt_arr.shape)


if __name__ == '__main__':
    main()
    