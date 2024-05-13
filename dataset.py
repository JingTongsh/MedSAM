import random
from os.path import join, isfile, basename
from glob import glob
from typing import Tuple
import numpy as np
from numpy import ndarray
import cv2
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate, hflip, vflip


def augment(image: Tensor, seg_label: Tensor = None) -> Tuple[Tensor, Tensor]:
    """
    Randomly augment the image and segmentation label together in the same way.
    :param image: (C, H, W)
    :param seg_label: (H, W)
    :return: augmented image and segmentation label
    """
        
    assert image.ndim == 3, 'image must be (C, H, W), but got {}'.format(image.shape)
    if seg_label is not None:
        assert seg_label.ndim == 2, 'seg_label must be (H, W), but got {}'.format(seg_label.shape)
        assert image.shape[1:] == seg_label.shape, 'image and seg_label must have the same spatial shape, but got {} and {}'.format(image.shape, seg_label.shape)
    
    seg_label = seg_label.unsqueeze(0)  # (1, H, W) for torchvision transforms
        
    prob = 0.5
    
    # rotate
    if torch.rand(1).item() < prob:
        k = torch.randint(0, 4, (1,)).item()
        image = rotate(image, k * 90)
        seg_label = rotate(seg_label, k * 90) if seg_label is not None else None

    # horizontal flip
    if torch.rand(1).item() < prob:
        image = hflip(image)
        seg_label = hflip(seg_label) if seg_label is not None else None
    
    # vertical flip
    if torch.rand(1).item() < prob:
        image = vflip(image)
        seg_label = vflip(seg_label) if seg_label is not None else None
    
    # random clip
    if torch.rand(1).item() < prob:
        h0, w0 = image.shape[-2:]
        h1, w1 = int(h0 * .8), int(w0 * .8)
        y_start = torch.randint(0, h0 - h1, (1,)).item()
        x_start = torch.randint(0, w0 - w1, (1,)).item()
        image = image[..., y_start:y_start+h1, x_start:x_start+w1]
        seg_label = seg_label[..., y_start:y_start+h1, x_start:x_start+w1] if seg_label is not None else None
        # resize to original size
        image = torch.nn.functional.interpolate(image.unsqueeze(0), (h0, w0), mode='bilinear', align_corners=False).squeeze(0)
        seg_label = torch.nn.functional.interpolate(seg_label.float().unsqueeze(0), (h0, w0), mode='nearest').squeeze(0).long() if seg_label is not None else None
    
    # add noise
    if torch.rand(1).item() < prob:
        noise = torch.randn_like(image) * 0.05
        image += noise
    
    seg_label = seg_label.squeeze(0)  # (H, W)

    return image, seg_label


class NpyDataset(Dataset): 
    def __init__(self, data_root, image_size=256, bbox_shift=10, data_aug=True):
        self.data_root = data_root
        self.gt_dir = join(data_root, 'gts')
        self.img_dir = join(data_root, 'imgs')
        self.gt_files = sorted(glob(join(self.gt_dir, '*.npy'), recursive=True))
        # keep gt with corresponding image
        self.gt_files = [file for file in self.gt_files if isfile(join(self.img_dir, basename(file)))]
        
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug
    
    def __len__(self):
        return len(self.gt_files)
    
    @staticmethod
    def resize_and_pad(data: ndarray, mode: int = cv2.INTER_AREA, normalize: bool = True) -> ndarray:
        """
        Resize the longest side to 256, then pad to (256, 256).
        :param data: input image, shape (H, W, C) or (H, W)
        :param mode: interpolation mode
        :param normalize: whether to normalize the image to [0, 1]
        :return: resized and padded image, shape (256, 256, C) or (256, 256)
        """
        assert data.ndim in [2, 3], 'data must be (H, W, C) or (H, W), but got {}'.format(data.shape)
        h, w = data.shape[:2]
        target_size = 256
        scale = target_size * 1.0 / max(h, w)
        newh, neww = int(h * scale + 0.5), int(w * scale + 0.5)
        data = cv2.resize(data, (neww, newh), interpolation=mode)
        
        if normalize:
            data = (data - data.min()) / np.clip(data.max() - data.min(), a_min=1e-8, a_max=None) # normalize to [0, 1]
        
        padh = target_size - newh
        padw = target_size - neww
        if data.ndim == 3:
            data = np.pad(data, ((0, padh), (0, padw), (0, 0)))
        else:
            data = np.pad(data, ((0, padh), (0, padw)))
        return data

    def __getitem__(self, index):
        # image
        img_name = basename(self.gt_files[index])
        img_file = join(self.img_dir, img_name)
        img = np.load(img_file, 'r', allow_pickle=True) # (H, W, 3)
        h1, w1 = img.shape[:2]
        img = self.resize_and_pad(img, cv2.INTER_AREA, True) # (256, 256, 3)
        img = np.transpose(img, (2, 0, 1)) # (3, 256, 256)
        assert np.max(img)<=1.0 and np.min(img)>=0.0, 'image should be normalized to [0, 1]'
        
        # gt
        gt = np.load(self.gt_files[index], 'r', allow_pickle=True) #   labels in [0, 1,4,5...]
        # assert gt.max() >= 1, 'gt should have at least one label'
        gt = self.resize_and_pad(gt, cv2.INTER_NEAREST, False) # (256, 256)
        # label_ids = np.unique(gt)[1:]
        gt2D = gt
        
        img = torch.from_numpy(img)
        gt2D = torch.from_numpy(gt2D)
        
        # data augmentation
        if self.data_aug:
            img, gt2D = augment(img, gt2D)
        
        # calculate bounding box from gt; deprecated
        gt2D = (gt2D > 0).to(torch.uint8)
        y_indices, x_indices = torch.where(gt2D > 0)
        x_min, x_max = torch.min(x_indices), torch.max(x_indices)
        y_min, y_max = torch.min(y_indices), torch.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = torch.tensor([[[x_min, y_min, x_max, y_max]]]).float()  # (1, 1, 4)
        
        return {
            "image": img.float(),
            "gt2D": gt2D.unsqueeze(0).long(), # (1, 256, 256)
            "bboxes": bboxes,
            "image_name": img_name,
            "new_size": torch.tensor([self.image_size, self.image_size]).long(),
            "original_size": torch.tensor([h1, w1]).long()
        }
