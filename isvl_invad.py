import argparse
import copy
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mvtec2_invad_dataset import build_invad_datasets
from utils import get_logger, setup_seed
from utils_invad import WarmupCosineScheduler, evaluation_batch_invad, export_private_maps, update_ema

from invad.backbones import get_backbone, get_backbone_feature_shape
from invad.denoiser import get_denoiser


def build_feature_extractor(args, device: str):
    backbone_kwargs = dict(
        model_type=args.backbone,
        outblocks=[1, 5, 9, 21],
        outstrides=[2, 4, 8, 16],
        pretrained=True,
        stride=16,
    )
    feature_extractor = get_backbone(**backbone_kwargs).to(device).eval()
    diff_in_sh = get_backbone_feature_shape(model_type=args.backbone)
    return feature_extractor, diff_in_sh


def build_denoiser(args, diff_in_sh: Tuple[int, int, int], device: str):
    model = get_denoiser(
        model_type=args.diff_model_type,
        num_classes=len(args.item_list),
        input_shape=diff_in_sh,
        z_channels=args.z_channels,
        depth=args.depth,
        width=args.width,
        num_sampling_steps=str(args.num_sampling_steps),
        grad_checkpoint=False,
        conditioning_scheme="none",
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        class_dropout_prob=0.0,
        learn_sigma=False,
        channel_mult=[1, 1, 2, 2],
    ).to(device)
    model_ema = copy.deepcopy(model).to(device)
    return model, model_ema


def train_one_epoch(model, model_ema, feature_extractor, train_loader, optimizer, scheduler, device, args):
    model.train()
    feature_extractor.eval()
    losses = []

    for batch in tqdm(train_loader, ncols=80):
        img = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)

        with torch.no_grad():
            x, _ = feature_extractor(img)  # (B, c, h, w)

        loss = model(x, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        update_ema(model_ema, model, decay=args.ema_decay)

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else 0.0


def main(args):
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    train_dataset, test_datasets, private_datasets, true_val_datasets = build_invad_datasets(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    feature_extractor, diff_in_sh = build_feature_extractor(args, device)
    model, model_ema = build_denoiser(args, diff_in_sh, device)

    print_fn(f"InvAD feature shape: {diff_in_sh}")
    print_fn(f"Using backbone: {args.backbone}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    total_steps = args.total_epochs * max(1, len(train_loader))
    warmup_steps = args.warmup_epochs * max(1, len(train_loader))
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        init_lr=args.init_lr,
        peak_lr=args.peak_lr,
        final_lr=args.final_lr,
    )

    ckpt_dir = os.path.join(args.save_dir, args.save_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_mean_auroc = -1.0

    if args.phase == "train":
        for epoch in range(args.total_epochs):
            loss = train_one_epoch(model, model_ema, feature_extractor, train_loader, optimizer, scheduler, device, args)
            print_fn(f"epoch [{epoch + 1}/{args.total_epochs}], loss: {loss:.6f}")

            if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.total_epochs:
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(args.item_list, test_datasets):
                    test_loader = DataLoader(
                        test_data,
                        batch_size=args.eval_batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )
                    results = evaluation_batch_invad(model_ema, feature_extractor, test_loader, device, args, diff_in_sh)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, _, _, eval_loss = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    print_fn(
                        f"{item}: eval_loss={eval_loss:.6f}, "
                        f"I-AUROC={auroc_sp:.4f}, I-AP={ap_sp:.4f}, I-F1={f1_sp:.4f}, "
                        f"P-AUROC={auroc_px:.4f}, P-AP={ap_px:.4f}, P-F1={f1_px:.4f}, P-AUPRO={aupro_px:.4f}"
                    )

                mean_auroc = float(np.mean(auroc_sp_list)) if auroc_sp_list else -1.0
                print_fn(
                    "Mean: I-AUROC={:.4f}, I-AP={:.4f}, I-F1={:.4f}, P-AUROC={:.4f}, P-AP={:.4f}, P-F1={:.4f}, P-AUPRO={:.4f}".format(
                        np.mean(auroc_sp_list),
                        np.mean(ap_sp_list),
                        np.mean(f1_sp_list),
                        np.mean(auroc_px_list),
                        np.mean(ap_px_list),
                        np.mean(f1_px_list),
                        np.mean(aupro_px_list),
                    )
                )

                torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_last.pth"))
                torch.save(model_ema.state_dict(), os.path.join(ckpt_dir, "model_ema_last.pth"))
                if mean_auroc > best_mean_auroc:
                    best_mean_auroc = mean_auroc
                    torch.save(model_ema.state_dict(), os.path.join(ckpt_dir, "model_ema_best.pth"))

    elif args.phase == "test":
        ckpt_path = os.path.join(ckpt_dir, args.ckpt_name)
        state = torch.load(ckpt_path, map_location=device)
        model_ema.load_state_dict(state, strict=True)
        model_ema.eval()

        auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
        auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

        for item, test_data in zip(args.item_list, test_datasets):
            test_loader = DataLoader(
                test_data,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            results = evaluation_batch_invad(model_ema, feature_extractor, test_loader, device, args, diff_in_sh)
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, _, _, eval_loss = results

            auroc_sp_list.append(auroc_sp)
            ap_sp_list.append(ap_sp)
            f1_sp_list.append(f1_sp)
            auroc_px_list.append(auroc_px)
            ap_px_list.append(ap_px)
            f1_px_list.append(f1_px)
            aupro_px_list.append(aupro_px)

            print_fn(
                f"{item}: eval_loss={eval_loss:.6f}, "
                f"I-AUROC={auroc_sp:.4f}, I-AP={ap_sp:.4f}, I-F1={f1_sp:.4f}, "
                f"P-AUROC={auroc_px:.4f}, P-AP={ap_px:.4f}, P-F1={f1_px:.4f}, P-AUPRO={aupro_px:.4f}"
            )

        print_fn(
            "Mean: I-AUROC={:.4f}, I-AP={:.4f}, I-F1={:.4f}, P-AUROC={:.4f}, P-AP={:.4f}, P-F1={:.4f}, P-AUPRO={:.4f}".format(
                np.mean(auroc_sp_list),
                np.mean(ap_sp_list),
                np.mean(f1_sp_list),
                np.mean(auroc_px_list),
                np.mean(ap_px_list),
                np.mean(f1_px_list),
                np.mean(aupro_px_list),
            )
        )

    elif args.phase == "val":
        ckpt_path = os.path.join(ckpt_dir, args.ckpt_name)
        state = torch.load(ckpt_path, map_location=device)
        model_ema.load_state_dict(state, strict=True)
        model_ema.eval()

        save_root = os.path.join(args.result_dir, "anomaly_images")
        os.makedirs(save_root, exist_ok=True)
        for item, private_data in zip(args.item_list, private_datasets):
            private_loader = DataLoader(
                private_data,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            export_private_maps(model_ema, feature_extractor, private_loader, device, args, diff_in_sh, save_root)
            print_fn(f"Saved anomaly maps for private split: {item}")

    else:
        raise ValueError(f"Unsupported phase: {args.phase}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./saved_results")
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--save_name", type=str, default="InvAD-MVTec2")
    parser.add_argument("--phase", type=str, default="train", choices=["train", "test", "val"])
    parser.add_argument("--ckpt_name", type=str, default="model_ema_best.pth")

    parser.add_argument("--item_list", nargs="+", default=["can"])
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--backbone", type=str, default="efficientnet-b4")
    parser.add_argument("--diff_model_type", type=str, default="dit")
    parser.add_argument("--z_channels", type=int, default=768)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--num_sampling_steps", type=str, default="100")
    parser.add_argument("--eval_step", type=int, default=3)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--total_epochs", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=25)
    parser.add_argument("--init_lr", type=float, default=1e-6)
    parser.add_argument("--peak_lr", type=float, default=5e-5)
    parser.add_argument("--final_lr", type=float, default=5e-6)
    parser.add_argument("--warmup_epochs", type=int, default=40)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.save_name = args.save_name + f"_Resize={args.input_size}_Backbone={args.backbone}_Classes={len(args.item_list)}"
    main(args)
