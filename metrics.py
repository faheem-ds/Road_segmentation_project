import torch
import numpy as np

from config import EDGE_METRIC_TOLERANCE
from losses import generate_edge_labels
from postprocessing import postprocess_binary_masks


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()
    fp = (preds * (1.0 - targets)).sum()
    fn = ((1.0 - preds) * targets).sum()
    tn = ((1.0 - preds) * (1.0 - targets)).sum()
    eps = 1e-7

    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "iou": iou.item(),
        "dice": dice.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "pixel_accuracy": pixel_accuracy.item(),
    }


def edge_iou_from_masks(preds: torch.Tensor, targets: torch.Tensor) -> float:
    pred_edges = generate_edge_labels(preds)
    target_edges = generate_edge_labels(targets)
    if EDGE_METRIC_TOLERANCE > 0:
        kernel_size = EDGE_METRIC_TOLERANCE * 2 + 1
        pred_match = torch.nn.functional.max_pool2d(pred_edges, kernel_size=kernel_size, stride=1, padding=EDGE_METRIC_TOLERANCE)
        target_match = torch.nn.functional.max_pool2d(target_edges, kernel_size=kernel_size, stride=1, padding=EDGE_METRIC_TOLERANCE)
        pred_true_positive = pred_edges * target_match
        target_true_positive = target_edges * pred_match
        intersection = pred_true_positive.sum() + target_true_positive.sum()
        union = pred_edges.sum() + target_edges.sum()
        return (intersection / (union + 1e-7)).item()

    pred_edges = pred_edges.view(-1)
    target_edges = target_edges.view(-1)
    intersection = (pred_edges * target_edges).sum()
    union = pred_edges.sum() + target_edges.sum() - intersection
    return (intersection / (union + 1e-7)).item()


def _connected_component_stats(mask_2d) -> tuple[int, float]:
    try:
        from skimage.measure import label
    except ImportError:
        label = None

    if label is None:
        mask = np.asarray(mask_2d, dtype=bool)
        visited = np.zeros(mask.shape, dtype=bool)
        component_count = 0
        largest_area = 0
        height, width = mask.shape

        for row in range(height):
            for col in range(width):
                if not mask[row, col] or visited[row, col]:
                    continue

                component_count += 1
                area = 0
                stack = [(row, col)]
                visited[row, col] = True

                while stack:
                    current_row, current_col = stack.pop()
                    area += 1
                    for next_row in range(max(0, current_row - 1), min(height, current_row + 2)):
                        for next_col in range(max(0, current_col - 1), min(width, current_col + 2)):
                            if mask[next_row, next_col] and not visited[next_row, next_col]:
                                visited[next_row, next_col] = True
                                stack.append((next_row, next_col))

                largest_area = max(largest_area, area)

        total_area = float(mask.sum())
        if component_count == 0 or total_area == 0.0:
            return component_count, 0.0
        return component_count, largest_area / total_area

    labeled = label(mask_2d, connectivity=2)
    component_count = int(labeled.max())
    total_area = float(mask_2d.sum())
    if component_count == 0 or total_area == 0.0:
        return component_count, 0.0

    largest_area = 0
    for component_id in range(1, component_count + 1):
        largest_area = max(largest_area, int((labeled == component_id).sum()))
    return component_count, largest_area / total_area


def connectivity_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    component_diffs = []
    largest_ratios = []

    for pred_mask, target_mask in zip(preds_np, targets_np):
        pred_2d = pred_mask.squeeze() > 0.5
        target_2d = target_mask.squeeze() > 0.5
        pred_components, largest_ratio = _connected_component_stats(pred_2d)
        target_components, _ = _connected_component_stats(target_2d)
        component_diffs.append(abs(pred_components - target_components))
        largest_ratios.append(largest_ratio)

    if not component_diffs:
        return {"connected_component_difference": 0.0, "largest_component_ratio": 0.0}
    return {
        "connected_component_difference": sum(component_diffs) / len(component_diffs),
        "largest_component_ratio": sum(largest_ratios) / len(largest_ratios),
    }


def full_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    apply_postprocessing: bool = False,
    include_connectivity: bool = True,
) -> dict[str, float]:
    metrics = segmentation_metrics(logits, targets, threshold)
    preds = (torch.sigmoid(logits) >= threshold).float()
    if apply_postprocessing:
        preds = postprocess_binary_masks(preds)
        metrics = segmentation_metrics(preds * 20.0 - 10.0, targets, threshold)
    metrics["edge_iou"] = edge_iou_from_masks(preds, targets.float())
    if include_connectivity:
        metrics.update(connectivity_metrics(preds, targets.float()))
    return metrics


def average_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {
            "iou": 0.0,
            "dice": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "pixel_accuracy": 0.0,
            "edge_iou": 0.0,
            "connected_component_difference": 0.0,
            "largest_component_ratio": 0.0,
        }
    keys = items[0].keys()
    return {key: sum(item[key] for item in items) / len(items) for key in keys}
