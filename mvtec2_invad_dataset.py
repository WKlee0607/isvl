import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from mvtec2_dataset import MVTec2Dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_invad_transforms(input_size: int):
    """InvAD uses ImageNet normalization and no data augmentation."""
    img_tf = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    gt_tf = transforms.Compose(
        [
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
    return img_tf, gt_tf


class InvADWrapper(Dataset):
    """Wrap the existing MVTec2Dataset into the dict format used by InvAD."""

    def __init__(self, base_dataset: Dataset, cls_name: str, cls_label: int, split: str):
        self.base = base_dataset
        self.cls_name = cls_name
        self.cls_label = cls_label
        self.split = split

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img, gt, label, img_path = self.base[idx]

        out = {
            "samples": img,
            "clsnames": self.cls_name,
            "clslabels": torch.tensor(self.cls_label, dtype=torch.long),
            "filenames": img_path,
        }

        if self.split != "train":
            out["labels"] = torch.tensor(label, dtype=torch.long)
            out["masks"] = gt

        return out


class InvADValWrapper(Dataset):
    """Private/test-time wrapper for exporting anomaly maps without GT labels."""

    def __init__(self, base_dataset: Dataset, cls_name: str, cls_label: int):
        self.base = base_dataset
        self.cls_name = cls_name
        self.cls_label = cls_label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img, gt, label, img_path = self.base[idx]
        return {
            "samples": img,
            "clsnames": self.cls_name,
            "clslabels": torch.tensor(self.cls_label, dtype=torch.long),
            "filenames": img_path,
        }


def build_invad_datasets(args):
    img_tf, gt_tf = build_invad_transforms(args.input_size)
    cls_to_idx = {name: i for i, name in enumerate(args.item_list)}

    train_sets: List[Dataset] = []
    test_sets: List[Dataset] = []
    private_sets: List[Dataset] = []
    true_val_sets: List[Dataset] = []

    for item in args.item_list:
        root = os.path.join(args.data_path, item)
        cls_idx = cls_to_idx[item]

        train_base = MVTec2Dataset(
            root=root,
            transform=img_tf,
            gt_transform=gt_tf,
            phase="train",
            resize=args.input_size,
            normal_only=True,
        )
        test_base = MVTec2Dataset(
            root=root,
            transform=img_tf,
            gt_transform=gt_tf,
            phase="test",
            resize=args.input_size,
            normal_only=False,
        )
        private_base = MVTec2Dataset(
            root=root,
            transform=img_tf,
            gt_transform=gt_tf,
            phase="val",
            resize=args.input_size,
            normal_only=False,
        )
        true_val_base = MVTec2Dataset(
            root=root,
            transform=img_tf,
            gt_transform=gt_tf,
            phase="true_val",
            resize=args.input_size,
            normal_only=False,
        )

        train_sets.append(InvADWrapper(train_base, item, cls_idx, "train"))
        test_sets.append(InvADWrapper(test_base, item, cls_idx, "test"))
        private_sets.append(InvADValWrapper(private_base, item, cls_idx))
        true_val_sets.append(InvADWrapper(true_val_base, item, cls_idx, "test"))

    train_dataset = ConcatDataset(train_sets)
    return train_dataset, test_sets, private_sets, true_val_sets
