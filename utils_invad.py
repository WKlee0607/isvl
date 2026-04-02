import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from invad.denoiser import get_denoiser
from utils import ader_evaluator


def update_ema(model_ema: torch.nn.Module, model: torch.nn.Module, decay: float = 0.999):
    with torch.no_grad():
        for p_ema, p in zip(model_ema.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        init_lr: float,
        peak_lr: float,
        final_lr: float,
    ):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(1, warmup_steps)
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.step_num = 0
        self._set_lr(init_lr)

    def _set_lr(self, lr: float):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            ratio = self.step_num / self.warmup_steps
            lr = self.init_lr + (self.peak_lr - self.init_lr) * ratio
        else:
            progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.final_lr + (self.peak_lr - self.final_lr) * cosine
        self._set_lr(lr)


@torch.no_grad()
def calculate_log_pdf(x: torch.Tensor) -> torch.Tensor:
    """Same image-level likelihood term used in InvAD evaluation."""
    ll = -0.5 * (x ** 2 + math.log(2 * math.pi))
    return ll.sum(dim=(1, 2, 3))


@torch.no_grad()
def build_eval_denoiser(train_model: torch.nn.Module, args, diff_in_sh: Tuple[int, int, int], device: str):
    eval_model = get_denoiser(
        model_type=args.diff_model_type,
        num_classes=len(args.item_list),
        input_shape=diff_in_sh,
        z_channels=args.z_channels,
        depth=args.depth,
        width=args.width,
        num_sampling_steps=str(args.eval_step),
        grad_checkpoint=False,
        conditioning_scheme="none",
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        class_dropout_prob=0.0,
        learn_sigma=False,
        channel_mult=[1, 1, 2, 2],
    ).to(device)
    eval_model.load_state_dict(train_model.state_dict(), strict=True)
    eval_model.eval()
    return eval_model


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


@torch.no_grad()
def evaluation_batch_invad(train_model, feature_extractor, dataloader, device, args, diff_in_sh):
    """
    InvAD-compatible evaluation:
    - features = feature_extractor(images)
    - z_T = ddim_reverse_sample(features, t=0)
    - pixel map = ||z_T||_2
    - image score = norm(diffs) + norm(nll)
    """
    eval_denoiser = build_eval_denoiser(train_model, args, diff_in_sh, device)

    feature_extractor.eval()
    eval_denoiser.eval()

    gt_list_px: List[np.ndarray] = []
    pr_list_px: List[np.ndarray] = []
    gt_list_sp: List[np.ndarray] = []
    diff_scores: List[np.ndarray] = []
    nll_scores: List[np.ndarray] = []
    anomaly_map_list: List[np.ndarray] = []
    gt_mask_list: List[np.ndarray] = []
    losses: List[float] = []

    for batch in tqdm(dataloader, ncols=80):
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        gt = batch["masks"].float()
        img_labels = batch["labels"]

        features, _ = feature_extractor(images)
        loss = train_model(features, labels)
        losses.append(float(loss.detach().cpu().mean().item()))

        start_t = torch.zeros(images.shape[0], device=device, dtype=torch.long)
        latents_last = eval_denoiser.ddim_reverse_sample(features, start_t, labels, eta=0.0)

        latents_last_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()  # (B, h, w)
        anomaly_map = F.interpolate(
            latents_last_l2.unsqueeze(1),
            size=(images.shape[2], images.shape[3]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        min_diffs_spatial = latents_last_l2.view(latents_last_l2.shape[0], -1).min(dim=1)[0]
        max_diffs_spatial = latents_last_l2.view(latents_last_l2.shape[0], -1).max(dim=1)[0]
        diffs = max_diffs_spatial - min_diffs_spatial
        nll = calculate_log_pdf(latents_last).mul(-1.0)

        gt = (gt > 0.5).float()
        if gt.ndim == 4 and gt.shape[1] > 1:
            gt = torch.max(gt, dim=1, keepdim=True)[0]

        gt_np = gt[:, 0].cpu().numpy().astype(np.uint8)
        amap_np = anomaly_map.cpu().numpy().astype(np.float32)

        gt_list_px.append(gt_np)
        pr_list_px.append(amap_np)
        gt_list_sp.append(img_labels.cpu().numpy().astype(np.uint8))
        diff_scores.append(diffs.cpu().numpy().astype(np.float32))
        nll_scores.append(nll.cpu().numpy().astype(np.float32))

        for i in range(amap_np.shape[0]):
            anomaly_map_list.append((np.clip(amap_np[i], 0, None) * 255.0).astype(np.uint8))
            gt_mask_list.append((gt_np[i] * 255).astype(np.uint8))

    gt_px = np.concatenate(gt_list_px, axis=0)
    pr_px = np.concatenate(pr_list_px, axis=0)
    gt_sp = np.concatenate(gt_list_sp, axis=0)
    diffs = np.concatenate(diff_scores, axis=0)
    nlls = np.concatenate(nll_scores, axis=0)
    pr_sp = _minmax(diffs) + _minmax(nlls)

    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_px, pr_sp, gt_px, gt_sp)
    return [
        auroc_sp,
        ap_sp,
        f1_sp,
        auroc_px,
        ap_px,
        f1_px,
        aupro_px,
        anomaly_map_list,
        gt_mask_list,
        float(np.mean(losses)) if losses else 0.0,
    ]


@torch.no_grad()
def export_private_maps(train_model, feature_extractor, dataloader, device, args, diff_in_sh, save_root: str):
    eval_denoiser = build_eval_denoiser(train_model, args, diff_in_sh, device)
    feature_extractor.eval()
    eval_denoiser.eval()

    for batch in tqdm(dataloader, ncols=80):
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        filenames = batch["filenames"]

        features, _ = feature_extractor(images)
        start_t = torch.zeros(images.shape[0], device=device, dtype=torch.long)
        latents_last = eval_denoiser.ddim_reverse_sample(features, start_t, labels, eta=0.0)
        latents_last_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
        anomaly_map = F.interpolate(
            latents_last_l2.unsqueeze(1),
            size=(images.shape[2], images.shape[3]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        anomaly_map = anomaly_map.cpu().numpy().astype(np.float32)

        for idx, img_path in enumerate(filenames):
            with Image.open(img_path) as im:
                w, h = im.size
            amap = cv2.resize(anomaly_map[idx], (w, h), interpolation=cv2.INTER_LINEAR)
            amap_uint8 = (np.clip(amap, 0, None) * 255.0).astype(np.uint8)

            rel = os.path.relpath(img_path, start=args.data_path)
            save_path = os.path.join(save_root, rel)
            save_path = os.path.splitext(save_path)[0] + ".tiff"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, amap_uint8)
