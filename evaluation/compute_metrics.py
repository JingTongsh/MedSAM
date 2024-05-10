import argparse
import glob
import os
import json
from tqdm import tqdm
import numpy as np
import cv2
import torch
from evaluation_with_torchmetrics import evaluate_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seg_dir', default='test_demo/segs', type=str)
    parser.add_argument('-g', '--gt_dir', default='test_demo/gts', type=str)
    parser.add_argument('-out_file', default='test_demo/metrics.json', type=str)
    args = parser.parse_args()

    seg_dir = args.seg_dir
    gt_dir = args.gt_dir
    out_file = args.out_file
    
    if not os.path.isdir(seg_dir):
        raise FileNotFoundError(f"Directory not found: {seg_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"Directory not found: {gt_dir}")
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    
    pred_npz_files = sorted(glob.glob(os.path.join(seg_dir, '*.npz')))
    gt_npz_files = sorted(glob.glob(os.path.join(gt_dir, '*.npz')))
    assert len(pred_npz_files) == len(gt_npz_files), f"Number of files mismatch: {len(pred_npz_files)} vs {len(gt_npz_files)}"
    
    metrics_dict = {}
    pred_all, gt_all = [], []
    
    for pred_file, gt_file in tqdm(zip(pred_npz_files, gt_npz_files), total=len(pred_npz_files)):
        base_name = os.path.basename(pred_file).split('.')[0]
        pred_npz = np.load(pred_file)
        gt_npz = np.load(gt_file)
        pred_arr = pred_npz['segs']  # (D, H, W)
        gt_arr = gt_npz['gts']  # (D, H, W)
        # if pred_arr.ndim == 2:
        #     pred_arr = pred_arr[np.newaxis, ...]  # (1, H, W)
        # if gt_arr.ndim == 2:
        #     gt_arr = gt_arr[np.newaxis, ...]  # (1, H, W)
        if pred_arr.shape != gt_arr.shape:
            print(f"Shape mismatch: {pred_file}, {gt_file}")
            print(pred_arr.shape, gt_arr.shape)
        metrics_single_pair = evaluate_metrics(pred_arr, gt_arr, task='binary')
        metrics_dict[base_name] = metrics_single_pair
        
        pred_arr = torch.from_numpy(pred_arr).unsqueeze(0).float()
        pred_resize = torch.nn.functional.interpolate(pred_arr, size=(256, 256), mode='nearest').squeeze(0).numpy()
        pred_all.append(pred_resize)
        gt_arr = torch.from_numpy(gt_arr).unsqueeze(0).float()
        gt_resize = torch.nn.functional.interpolate(gt_arr, size=(256, 256), mode='nearest').squeeze(0).numpy()
        gt_all.append(gt_resize)
    
    pred_all = np.concatenate(pred_all, axis=0)
    gt_all = np.concatenate(gt_all, axis=0)
    metrics_overall = evaluate_metrics(pred_all, gt_all, task='binary')
    metrics_dict['overall'] = metrics_overall
    
    with open(out_file, 'w') as f:
        json.dump(metrics_dict, f, indent=4)


if __name__ == '__main__':
    main()
    