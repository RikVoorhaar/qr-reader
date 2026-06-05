"""Clustering candidate alignment patterns into merged clusters."""

from typing import NamedTuple

import numpy as np


class CandidateCluster(NamedTuple):
    row: float
    cols: np.ndarray  # shape (6,): [left_outer, left_inner, left_center, right_center, right_inner, right_outer]
    height: float
    num_candidates: int


def candidate_length(candidate: CandidateCluster) -> float:
    # Compute the full width of the candidate using the outermost column boundaries (cols[0] = left_outer, cols[5] = right_outer).
    return float(candidate.cols[5] - candidate.cols[0])


def candidate_lengths_match(
    candidate1: CandidateCluster,
    candidate2: CandidateCluster,
    length_thresh: float = 0.20,
) -> bool:
    """Check if the log-length difference of two candidates is within length_thresh."""

    log_length1 = np.log(candidate_length(candidate1))
    log_length2 = np.log(candidate_length(candidate2))
    return bool(np.abs(log_length1 - log_length2) < length_thresh)


def candidate_overlaps(
    candidate1: CandidateCluster,
    candidate2: CandidateCluster,
    length_thresh: float = 0.20,
    dist_thresh: float = 1.20,
) -> bool:
    """Check if two candidate clusters overlap in both the x and y dimensions.

    Two candidates are considered overlapping when all of the following hold:
    - Their lengths are similar (log-length difference within length_thresh).
    - Their row positions are within dist_thresh * candidate1.height of each other.
    - The horizontal center of candidate2 falls within the inner column span
      (cols[2] to cols[3]) of candidate1.
    """
    length_match = candidate_lengths_match(candidate1, candidate2, length_thresh)
    y_match = np.abs(candidate1.row - candidate2.row) < dist_thresh * candidate1.height
    candidate2_center = (candidate2.cols[2] + candidate2.cols[3]) / 2
    x_match = (candidate1.cols[2] <= candidate2_center) & (
        candidate2_center <= candidate1.cols[3]
    )
    return bool(length_match & y_match & x_match)


def merge_candidates(
    candidate1: CandidateCluster, candidate2: CandidateCluster
) -> CandidateCluster:
    """Merge two candidate clusters into one by computing weighted averages."""

    num_candidates = candidate1.num_candidates + candidate2.num_candidates
    row = (
        candidate1.row * candidate1.num_candidates
        + candidate2.row * candidate2.num_candidates
    ) / num_candidates
    cols = (
        candidate1.cols * candidate1.num_candidates
        + candidate2.cols * candidate2.num_candidates
    ) / num_candidates
    height = (
        candidate1.height * candidate1.num_candidates
        + candidate2.height * candidate2.num_candidates
    ) / num_candidates
    return CandidateCluster(row, cols, height, num_candidates)


def _choose_ref_candidate(
    candidates: CandidateCluster,
    processed_mask: np.ndarray,
    rng: np.random.Generator,
):
    """Pick a random unprocessed candidate as the reference for a new cluster."""
    unprocessed = np.where(~processed_mask)[0]
    index = rng.choice(unprocessed)
    ref_candidate = CandidateCluster(
        candidates.row[index],
        candidates.cols[index],
        candidates.height[index],
        candidates.num_candidates[index],
    )
    processed_mask = processed_mask.copy()
    processed_mask[index] = True
    return ref_candidate, processed_mask, rng


def cluster_candidates(rows: np.ndarray, cols_all: np.ndarray) -> list[CandidateCluster]:
    """
    Cluster raw alignment pattern candidates by merging overlapping ones.

    Parameters
    ----------
    rows : np.ndarray, shape (N,)
        Row indices of the candidate patterns.
    cols_all : np.ndarray, shape (N, 6)
        Six column boundaries per candidate.

    Returns
    -------
    list of CandidateCluster
    """
    rows_cand = np.array(rows, dtype=np.float32)
    cols_cand = np.array(cols_all, dtype=np.float32)
    height_cand = np.array(cols_cand[:, 3] - cols_cand[:, 2], dtype=np.float32)

    candidates = CandidateCluster(
        rows_cand, cols_cand, height_cand, np.ones(rows_cand.shape, dtype=np.int32)
    )
    num_candidates = candidates.num_candidates.shape[0]
    rng = np.random.default_rng(0)
    processed_mask = np.zeros(num_candidates, dtype=bool)

    clusters: list[CandidateCluster] = []
    while not processed_mask.all():
        ref_candidate, processed_mask, rng = _choose_ref_candidate(
            candidates, processed_mask, rng
        )
        for i in range(num_candidates):
            if processed_mask[i]:
                continue
            cand_i = CandidateCluster(
                candidates.row[i],
                candidates.cols[i],
                candidates.height[i],
                candidates.num_candidates[i],
            )
            if candidate_overlaps(ref_candidate, cand_i):
                ref_candidate = merge_candidates(ref_candidate, cand_i)
                processed_mask[i] = True
        clusters.append(ref_candidate)

    return clusters
