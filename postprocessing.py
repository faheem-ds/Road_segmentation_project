from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    APPLY_POSTPROCESSING,
    POSTPROCESS_CLOSING_ITERATIONS,
    POSTPROCESS_CLOSING_KERNEL,
    POSTPROCESS_MIN_COMPONENT_SIZE,
)


def morphological_close_masks(masks: torch.Tensor, kernel_size: int = POSTPROCESS_CLOSING_KERNEL, iterations: int = POSTPROCESS_CLOSING_ITERATIONS) -> torch.Tensor:
    if kernel_size <= 1 or iterations <= 0:
        return masks

    padding = kernel_size // 2
    closed = masks.float()
    for _ in range(iterations):
        closed = F.max_pool2d(closed, kernel_size=kernel_size, stride=1, padding=padding)
        closed = -F.max_pool2d(-closed, kernel_size=kernel_size, stride=1, padding=padding)
    return (closed > 0.5).float()


def _remove_small_components_np(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return mask.astype(np.uint8)

    mask = mask.astype(bool)
    visited = np.zeros(mask.shape, dtype=bool)
    output = np.zeros(mask.shape, dtype=np.uint8)
    height, width = mask.shape

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue

            component = []
            queue = deque([(row, col)])
            visited[row, col] = True

            while queue:
                current_row, current_col = queue.popleft()
                component.append((current_row, current_col))
                for next_row in range(max(0, current_row - 1), min(height, current_row + 2)):
                    for next_col in range(max(0, current_col - 1), min(width, current_col + 2)):
                        if mask[next_row, next_col] and not visited[next_row, next_col]:
                            visited[next_row, next_col] = True
                            queue.append((next_row, next_col))

            if len(component) >= min_size:
                for component_row, component_col in component:
                    output[component_row, component_col] = 1

    return output


def remove_small_components(masks: torch.Tensor, min_size: int = POSTPROCESS_MIN_COMPONENT_SIZE) -> torch.Tensor:
    if min_size <= 1:
        return masks.float()

    cleaned = []
    for mask in masks.detach().cpu().numpy():
        cleaned_mask = _remove_small_components_np(mask.squeeze() > 0.5, min_size)
        cleaned.append(torch.from_numpy(cleaned_mask).unsqueeze(0))
    return torch.stack(cleaned, dim=0).to(device=masks.device, dtype=torch.float32)


def postprocess_binary_masks(masks: torch.Tensor) -> torch.Tensor:
    if not APPLY_POSTPROCESSING:
        return masks.float()

    masks = morphological_close_masks(masks)
    masks = remove_small_components(masks)
    return masks.float()
