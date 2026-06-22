import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CLDICE_ITERATIONS, EDGE_KERNEL_SIZE, EDGE_LOSS_MAX_POS_WEIGHT, LAMBDA_CENTERLINE, LAMBDA_CLDICE


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice(logits, targets)


def generate_edge_labels(masks: torch.Tensor, kernel_size: int = EDGE_KERNEL_SIZE) -> torch.Tensor:
    """Morphological gradient edge labels from binary masks."""
    masks = (masks > 0.5).float()
    padding = kernel_size // 2
    dilated = F.max_pool2d(masks, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-masks, kernel_size=kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


def balanced_edge_bce_loss(edge_logits: torch.Tensor, edge_targets: torch.Tensor) -> torch.Tensor:
    positive = edge_targets.sum()
    negative = edge_targets.numel() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(max=EDGE_LOSS_MAX_POS_WEIGHT)
    return F.binary_cross_entropy_with_logits(edge_logits, edge_targets, pos_weight=pos_weight)


def balanced_bce_loss(logits: torch.Tensor, targets: torch.Tensor, max_pos_weight: float = EDGE_LOSS_MAX_POS_WEIGHT) -> torch.Tensor:
    positive = targets.sum()
    negative = targets.numel() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(max=max_pos_weight)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def soft_erode(mask: torch.Tensor) -> torch.Tensor:
    erode_horizontal = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
    erode_vertical = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(erode_horizontal, erode_vertical)


def soft_dilate(mask: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)


def soft_open(mask: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(mask))


def soft_skeletonize(mask: torch.Tensor, iterations: int = CLDICE_ITERATIONS) -> torch.Tensor:
    skeleton = F.relu(mask - soft_open(mask))
    for _ in range(iterations):
        mask = soft_erode(mask)
        opened = soft_open(mask)
        delta = F.relu(mask - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0.0, 1.0)


def soft_cldice_loss(logits: torch.Tensor, targets: torch.Tensor, iterations: int = CLDICE_ITERATIONS) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = (targets > 0.5).float()
    pred_skeleton = soft_skeletonize(probs, iterations)
    target_skeleton = soft_skeletonize(targets, iterations)
    eps = 1e-7

    topology_precision = (pred_skeleton * targets).sum(dim=(1, 2, 3)) / (pred_skeleton.sum(dim=(1, 2, 3)) + eps)
    topology_recall = (target_skeleton * probs).sum(dim=(1, 2, 3)) / (target_skeleton.sum(dim=(1, 2, 3)) + eps)
    cldice = (2.0 * topology_precision * topology_recall) / (topology_precision + topology_recall + eps)
    return 1.0 - cldice.mean()


class EdgeAwareLoss(nn.Module):
    def __init__(self, lambda_edge: float = 0.3):
        super().__init__()
        self.segmentation_loss = BCEDiceLoss()
        self.lambda_edge = lambda_edge

    def forward(self, outputs: dict[str, torch.Tensor], masks: torch.Tensor) -> torch.Tensor:
        seg_logits = outputs["seg_logits"]
        edge_logits = outputs["edge_logits"]
        edge_targets = generate_edge_labels(masks)
        centerline_targets = soft_skeletonize((masks > 0.5).float()).detach()
        seg_loss = self.segmentation_loss(seg_logits, masks)
        edge_loss = balanced_edge_bce_loss(edge_logits, edge_targets)
        cldice_loss = soft_cldice_loss(seg_logits, masks)
        centerline_loss = torch.tensor(0.0, device=masks.device)
        if "centerline_logits" in outputs:
            centerline_loss = balanced_bce_loss(outputs["centerline_logits"], centerline_targets)
        return (
            seg_loss
            + self.lambda_edge * edge_loss
            + LAMBDA_CLDICE * cldice_loss
            + LAMBDA_CENTERLINE * centerline_loss
        )
