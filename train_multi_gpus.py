import argparse
import random
import os
from os import listdir, makedirs
from os.path import join, isfile
from copy import deepcopy
from time import time
from shutil import copyfile
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch import multiprocessing as mp
from torch import distributed as dist
from datetime import datetime
from matplotlib import pyplot as plt
from monai.losses import DiceLoss

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
from LiteMedSAM import MedSAM_Lite
from dataset import NpyDataset
from evaluation import evaluate_metrics


torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--tr_npy_path', type=str,
                        default='prostate_foreground/tr_npy',
                        help='Path to training npy files; two subfolders: gts and imgs')
    parser.add_argument('-v', '--ts_npy_path', type=str,
                        default='prostate_foreground/ts_npy',
                        help='Path to testing npy files; two subfolders: gts and imgs')
    parser.add_argument('-task_name', type=str, default='MedSAM-Lite')
    parser.add_argument('-pretrained_checkpoint', type=str, default='lite_medsam.pth',
                        help='Path to pretrained MedSAM-Lite checkpoint')
    parser.add_argument('-work_dir', type=str, default='./work_dir')
    parser.add_argument('--data_aug', action='store_true', default=False,
                        help='use data augmentation during training')
    # train
    parser.add_argument('-num_epochs', type=int, default=1000)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-num_workers', type=int, default=8)
    # Optimizer parameters
    parser.add_argument('-weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('-lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (absolute lr)')
    ## Distributed training args
    parser.add_argument('-world_size', type=int, help='world size')
    parser.add_argument('-node_rank', type=int, help='Node rank')
    parser.add_argument('-bucket_cap_mb', type = int, default = 25,
                        help='The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)')
    parser.add_argument('-resume', type = str, default = '', required=False,
                        help="Resuming training from a work_dir")
    parser.add_argument('-init_method', type = str, default = "env://")
    args = parser.parse_args()

    return args


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.45])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))


@torch.no_grad()
def cal_iou(result, reference):
    
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
    
    invalid = union == 0
    iou = intersection.float() / union.float()
    iou[invalid] = 0.0
    return iou.unsqueeze(1)


def revert_sync_batchnorm(module: torch.nn.Module) -> torch.nn.Module:
    # Code adapted from https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547
    # Original author: Kapil Yedidi (@kapily)
    converted_module = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        # Unfortunately, SyncBatchNorm does not store the original class - if it did
        # we could return the one that was originally created.
        converted_module = nn.BatchNorm2d(
            module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats
        )
        if module.affine:
            with torch.no_grad():
                converted_module.weight = module.weight
                converted_module.bias = module.bias
        converted_module.running_mean = module.running_mean
        converted_module.running_var = module.running_var
        converted_module.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            converted_module.qconfig = module.qconfig
    for name, child in module.named_children():
        converted_module.add_module(name, revert_sync_batchnorm(child))
    del module

    return converted_module




def collate_fn(batch):
    """
    Collate function for PyTorch DataLoader.
    """
    batch_dict = {}
    for key in batch[0].keys():
        if key == "image_name":
            batch_dict[key] = [sample[key] for sample in batch]
        else:
            batch_dict[key] = torch.stack([sample[key] for sample in batch], dim=0)

    return batch_dict

#%% sanity test of dataset class
def sanity_check_dataset(args):
    print('tr_npy_path', args.tr_npy_path)
    tr_dataset = NpyDataset(args.tr_npy_path, data_aug=args.data_aug)
    print('len(tr_dataset)', len(tr_dataset))
    tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    makedirs(args.work_dir, exist_ok=True)
    for step, batch in enumerate(tr_dataloader):
        # print(image.shape, gt.shape, bboxes.shape)
        # show the example
        _, axs = plt.subplots(1, 2, figsize=(10, 10))
        idx = random.randint(0, 4)

        image = batch["image"]
        gt = batch["gt2D"]
        bboxes = batch["bboxes"]
        names_temp = batch["image_name"]

        axs[0].imshow(image[idx].cpu().permute(1,2,0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[0])
        show_box(bboxes[idx].numpy().squeeze(), axs[0])
        axs[0].axis('off')
        # set title
        axs[0].set_title(names_temp[idx])
        idx = random.randint(4, 7)
        axs[1].imshow(image[idx].cpu().permute(1,2,0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[1])
        show_box(bboxes[idx].numpy().squeeze(), axs[1])
        axs[1].axis('off')
        # set title
        axs[1].set_title(names_temp[idx])
        # plt.show()  
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig(
            join(args.work_dir, 'medsam_lite-train_bbox_prompt_sanitycheck_DA.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        break


def main(args):
    ngpus_per_node = torch.cuda.device_count()
    print("Spawning processes")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = args.world_size
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    if is_main_host:
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
        makedirs(model_save_path, exist_ok=True)
        copyfile(
            __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
        )
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:{}".format(gpu))
    dist.init_process_group(
        backend="nccl", init_method=args.init_method, rank=rank, world_size=world_size
    )

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    medsam_lite_image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[
            64, ## (64, 256, 256)
            128, ## (128, 128, 128)
            160, ## (160, 64, 64)
            320 ## (320, 64, 64) 
        ],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8
    )

    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )

    medsam_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
    )
    
    medsam_lite_model = MedSAM_Lite(
        image_encoder = medsam_lite_image_encoder,
        mask_decoder = medsam_lite_mask_decoder,
        prompt_encoder = medsam_lite_prompt_encoder
    )
    
    if (not os.path.exists(args.resume)) and isfile(args.pretrained_checkpoint):
        ## Load pretrained checkpoint if there's no checkpoint to resume from and there's a pretrained checkpoint
        print(f"Loading pretrained checkpoint from {args.pretrained_checkpoint}")
        medsam_lite_checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu")
        medsam_lite_model.load_state_dict(medsam_lite_checkpoint, strict=False)

    medsam_lite_model = medsam_lite_model.to(device)

    ## Make sure there's only 2d BN layers, so that I can revert them properly
    for module in medsam_lite_model.modules():
        cls_name = module.__class__.__name__
        if "BatchNorm" in cls_name:
            assert cls_name == "BatchNorm2d" 
    medsam_lite_model = nn.SyncBatchNorm.convert_sync_batchnorm(medsam_lite_model)

    medsam_lite_model = nn.parallel.DistributedDataParallel(
        medsam_lite_model,
        device_ids=[gpu],
        output_device=gpu,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb
    )
    medsam_lite_model.train()
    print(f"MedSAM Lite size: {sum(p.numel() for p in medsam_lite_model.parameters())}")
    # fix image encoder and prompter encoder, train mask decoder
    medsam_lite_model.module.image_encoder.requires_grad_(False)
    medsam_lite_model.module.image_encoder.additional_convs.requires_grad_(True)
    medsam_lite_model.module.image_encoder.conv_down.requires_grad_(True)
    medsam_lite_model.module.prompt_encoder.requires_grad_(False)
    optimizer = optim.AdamW(
        medsam_lite_model.module.mask_decoder.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=5,
        cooldown=0
    )
    seg_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    iou_loss = nn.MSELoss(reduction='mean')
    tr_data_root = args.tr_npy_path
    train_dataset = NpyDataset(data_root=tr_data_root, data_aug=True)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=collate_fn
    )
    ts_data_root = args.ts_npy_path
    test_dataset = NpyDataset(data_root=ts_data_root, data_aug=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    if os.path.exists(args.resume):
        ckpt_folders = sorted(listdir(args.resume))
        ckpt_folders = [f for f in ckpt_folders if (f.startswith(args.task_name) and isfile(join(args.resume, f, 'medsam_lite_latest.pth')))]
        print('*'*20)
        print('existing ckpts in', args.resume, ckpt_folders)
        # find the latest ckpt folders
        time_strings = [f.split(args.task_name + '-')[-1] for f in ckpt_folders]
        dates = [datetime.strptime(f, '%Y%m%d-%H%M') for f in time_strings]
        latest_date = max(dates)
        latest_ckpt = join(args.work_dir, args.task_name + '-' + latest_date.strftime('%Y%m%d-%H%M'), 'medsam_lite_latest.pth')
        print('Loading from', latest_ckpt)
        checkpoint = torch.load(latest_ckpt, map_location=device)
        medsam_lite_model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_loss = 1e10

    train_losses = []
    test_losses = []
    epoch_times = []
    for epoch in range(start_epoch, num_epochs):
        # train model
        medsam_lite_model.train()
        epoch_loss = [1e10 for _ in range(len(train_loader))]
        epoch_start_time = time()
        pbar = tqdm(train_loader)
        for step, batch in enumerate(pbar):
            image = batch["image"]
            gt2D = batch["gt2D"]
            boxes = batch["bboxes"]
            optimizer.zero_grad()
            image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
            logits_pred, iou_pred = medsam_lite_model(image, boxes)
            l_seg = seg_loss(logits_pred, gt2D)
            l_ce = ce_loss(logits_pred, gt2D.float())
            mask_loss = l_seg + l_ce
            with torch.no_grad():
                iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
            l_iou = iou_loss(iou_pred, iou_gt)
            loss = mask_loss + l_iou
            epoch_loss[step] = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"[RANK {rank}, TRAINING] Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

        epoch_end_time = time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        epoch_loss_world = [None for _ in range(world_size)]
        dist.all_gather_object(epoch_loss_world, epoch_loss)
        epoch_loss_reduced = np.vstack(epoch_loss_world).mean()
        train_losses.append(epoch_loss_reduced)
        lr_scheduler.step(epoch_loss_reduced)

        if is_main_host:
            module_revert_sync_BN = revert_sync_batchnorm(deepcopy(medsam_lite_model.module))
            weights = module_revert_sync_BN.state_dict()
            checkpoint = {
                "model": weights,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "loss": epoch_loss_reduced,
                "best_loss": best_loss,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_lite_latest.pth"))
        if epoch_loss_reduced < best_loss:
            print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
            best_loss = epoch_loss_reduced
            if is_main_host:
                checkpoint["best_loss"] = best_loss
                torch.save(checkpoint, join(model_save_path, "medsam_lite_best.pth"))
        dist.barrier()
        epoch_loss_reduced = 1e10
        # plot loss
        if is_main_host:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            axes[0].title.set_text("Dice + Binary Cross Entropy + IoU Loss")
            axes[0].plot(train_losses)
            axes[0].set_ylabel("Loss")
            axes[1].plot(epoch_times)
            axes[1].title.set_text("Epoch Duration")
            axes[1].set_ylabel("Duration (s)")
            axes[1].set_xlabel("Epoch")
            plt.tight_layout()
            plt.savefig(join(model_save_path, "train-loss.png"))
            plt.close()
        dist.barrier()
        
        # evaluate with test set
        medsam_lite_model.eval()
        test_loss = []
        logits_pred_all, logits_gt_all = [], []
        # iou_pred_all, iou_gt_all = [], []
        pbar = tqdm(test_loader)
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                image = batch["image"]  # (B, 3, 256, 256)
                gt2D = batch["gt2D"]  # (B, 1, 256, 256)
                boxes = batch["bboxes"]
                image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
                logits_pred, iou_pred = medsam_lite_model(image, boxes)  # (B, 1, 256, 256), (B, 1)
                
                l_seg = seg_loss(logits_pred, gt2D)
                l_ce = ce_loss(logits_pred, gt2D.float())
                mask_loss = l_seg + l_ce
                iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())  # (B, 1)
                l_iou = iou_loss(iou_pred, iou_gt)
                loss = mask_loss + l_iou
                test_loss.append(loss.item())
                
                logits_pred_all.append(logits_pred.detach().cpu())
                logits_gt_all.append(gt2D.detach().cpu())
                # iou_pred_all.append(iou_pred.detach().cpu())
                # iou_gt_all.append(iou_gt.detach().cpu())
                
                pbar.set_description(f"[RANK {rank}, TESTING] Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")
                
        test_loss = np.mean(test_loss)
        test_losses.append(test_loss)
        
        if is_main_host:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.plot(test_losses)
            ax.set_title("Test Loss")
            ax.set_ylabel("Loss")
            ax.set_xlabel("Epoch")
            plt.tight_layout()
            plt.savefig(join(model_save_path, "test-loss.png"))
            plt.close()
            
            logits_pred_all = torch.cat(logits_pred_all, dim=0)  # (N, 1, 256, 256)
            logits_gt_all = torch.cat(logits_gt_all, dim=0)  # (N, 1, 256, 256)
            
            metrics = evaluate_metrics(logits_pred_all, logits_gt_all, task='binary')
            print(metrics)
            with open(join(model_save_path, "metrics.txt"), "a") as f:
                f.write(f"Epoch {epoch}\n")
                f.write(str(metrics))
                f.write("\n")
            
        
if __name__ == "__main__":
    args = get_args()
    sanity_check_dataset(args)
    main(args)
