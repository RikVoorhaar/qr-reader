"""Region filling, boundary extraction, and connected components."""

from collections import defaultdict

import networkx as nx
import numpy as np
from scipy import ndimage


def get_neighbors(pixel: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    """Return 8-connected neighbors of a pixel within image bounds."""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            neighbor = (pixel[0] + dy, pixel[1] + dx)
            if 0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1]:
                neighbors.append(neighbor)
    return neighbors


def expand_wave_front_neighbors(wf: np.ndarray) -> np.ndarray:
    """OR of wf shifted so each True pixel contributes all 8 neighbors."""
    out = np.zeros_like(wf)
    out[:-1, :] = wf[1:, :]
    out[1:, :] |= wf[:-1, :]
    out[:, :-1] |= wf[:, 1:]
    out[:, 1:] |= wf[:, :-1]
    out[:-1, :-1] |= wf[1:, 1:]
    out[:-1, 1:] |= wf[1:, :-1]
    out[1:, :-1] |= wf[:-1, 1:]
    out[1:, 1:] |= wf[:-1, :-1]
    return out


def region_fill_wave_front(
    img_binary: np.ndarray,
    seed_row: int,
    seed_col: int,
) -> np.ndarray:
    """
    Connected region fill (8-neighbor) matching the seed pixel value.
    Uses pure Python while loop and numpy vectorized operations.
    """
    img = np.asarray(img_binary)
    target = img[seed_row, seed_col]
    region_mask = np.zeros_like(img, dtype=bool)
    wave_front = np.zeros_like(img, dtype=bool)
    wave_front[seed_row, seed_col] = True
    region_mask[seed_row, seed_col] = True

    while np.any(wave_front):
        expanded = expand_wave_front_neighbors(wave_front)
        new_pixels = expanded & (img == target) & (~region_mask)
        region_mask |= new_pixels
        wave_front = new_pixels

    return region_mask


def region_boundary_8(region_mask: np.ndarray) -> np.ndarray:
    """In-region pixels with at least one 8-neighbor not in the region."""
    return region_mask & expand_wave_front_neighbors(~region_mask)


def boundary_connected_components_networkx(
    boundary_mask: np.ndarray,
) -> list[list[tuple[int, int]]]:
    """
    8-connected components among True boundary pixels (NetworkX).
    """
    boundary_mask = np.asarray(boundary_mask, dtype=bool)
    h, w = boundary_mask.shape
    g = nx.Graph()
    for y, x in zip(*np.where(boundary_mask), strict=True):
        g.add_node((int(y), int(x)))
    for y, x in zip(*np.where(boundary_mask), strict=True):
        i = y * w + x
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx_ < w and boundary_mask[ny, nx_]:
                    j = ny * w + nx_
                    if j > i:
                        g.add_edge((int(y), int(x)), (int(ny), int(nx_)))
    return [sorted(c) for c in nx.connected_components(g)]


def boundary_connected_components_ndimage(
    boundary_mask: np.ndarray,
) -> list[list[tuple[int, int]]]:
    """
    Same 8-connected components using ``scipy.ndimage.label`` (C implementation).
    """
    boundary_mask = np.asarray(boundary_mask, dtype=bool)
    structure = ndimage.generate_binary_structure(2, 2)
    labeled, _n = ndimage.label(boundary_mask, structure=structure)
    by_label: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for y, x in zip(*np.where(boundary_mask), strict=True):
        by_label[int(labeled[y, x])].append((int(y), int(x)))
    return [sorted(v) for _, v in sorted(by_label.items())]
