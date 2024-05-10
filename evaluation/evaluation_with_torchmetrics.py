
from typing import Dict, Literal
from numpy import ndarray
import torch
from torch import Tensor
import torchmetrics
import os


def precision_recall_to_iou(p: float, r: float) -> float:
    if p * r == 0:
        return 0
    else:
        return p * r / (p + r - p * r)


def precision_recall_to_f1(p: float, r: float) -> float:
    if p * r == 0:
        return 0
    else:
        return 2 * p * r / (p + r)


def evaluate_metrics(
    y_pred: Tensor | ndarray,
    y_true: Tensor | ndarray,
    thresh: float = .5,
    task: Literal['binary', 'multiclass', 'multilabel'] = 'multiclass',
    num_classes: int = None,
    target_categories: Dict[int, str] = None
) -> Dict[str, float]:
    """
    Evaluate classification or segmentation metrics with CUDA device.
    :param y_pred: (N, *) labels or (N, C, *) logits
    :param y_true: (N, *) labels not one-hot, where * is the same as y_pred
    :param thresh: threshold for binary classification
    :param task: 'binary', 'multiclass', or 'multilabel'
    :param num_classes: number of classes, only used when task is 'multilabel'
    :param target_categories: dictionary of class labels, used in 'multiclass' task to show class names
    :return: dictionary of metrics
    """
    if task == 'multiclass':
        assert num_classes > 1, f'invalid number of classes: {num_classes}'
    # check shape
    assert y_pred.ndim - y_true.ndim in [0, 1], f'invalid dimensions: {y_pred.ndim}, {y_true.ndim}'
    
    # print('Evaluating metrics ...')
    
    if isinstance(y_pred, ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, ndarray):
        y_true = torch.from_numpy(y_true)
    
    if y_pred.ndim == y_true.ndim + 1:
        y_pred = torch.argmax(y_pred, dim=1)  # (N, C, *) -> (N, *)
    else:
        y_pred = (y_pred > thresh).float()  # (N, *) logits -> (N, *) labels
    
    assert y_pred.shape == y_true.shape, f'shape mismatch: {y_pred.shape}, {y_true.shape}'

    device = y_pred.device
    if 'cuda' not in str(device):
        device = int(os.environ.get('LOCAL_RANK')) if 'LOCAL_RANK' in os.environ else 0
    
    y_pred, y_true = y_pred.to(device), y_true.to(device)

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    acc = torchmetrics.functional.accuracy(y_pred, y_true, task=task, num_classes=num_classes)
    metrics = {'acc': acc.item()}
    
    prec = torchmetrics.functional.precision(y_pred, y_true, task=task, num_classes=num_classes, average='none')
    rec = torchmetrics.functional.recall(y_pred, y_true, task=task, num_classes=num_classes, average='none')
    
    if task == 'binary':
        metrics['precision'] = prec.item()
        metrics['recall'] = rec.item()
        metrics['f1'] = precision_recall_to_f1(prec.item(), rec.item())
        metrics['iou'] = precision_recall_to_iou(prec.item(), rec.item())
    elif task == 'multiclass':
        if target_categories is None:
            target_categories = {k: str(k) for k in range(num_classes)}
        for k, name in target_categories.items():
            metrics[f'precision_{name}'] = prec[k].item()
            metrics[f'recall_{name}'] = rec[k].item()
            metrics[f'f1_{name}'] = precision_recall_to_f1(prec[k].item(), rec[k].item())
            metrics[f'iou_{name}'] = precision_recall_to_iou(prec[k].item(), rec[k].item())

    return metrics


if __name__ == '__main__':
    # test
    device = 'cuda'
    pred = torch.rand(2, 3, 224, 224, device=device)
    target = torch.randint(0, 3, (2, 224, 224), device=device)
    target_categories = {1: 'object1', 2: 'object2'}
    metrics = evaluate_metrics(pred, target, target_categories=target_categories)
    print(metrics)
