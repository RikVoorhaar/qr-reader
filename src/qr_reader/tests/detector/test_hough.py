"""Tests for detector/hough.py — gradient-guided Hough voting and line refinement."""

import numpy as np
import pytest

from qr_reader.detector.hough import (
    LineSegment,
    hough_vote_peaks,
    refine_line,
)

# ---------------------------------------------------------------------------
# hough_vote_peaks
# ---------------------------------------------------------------------------


def _synthetic_edges_horizontal(
    shape=(50, 50),
    n_edges=4,
    y_start=20,
) -> tuple[np.ndarray, np.ndarray]:
    """Create (nms, angle) with horizontal edges.

    A horizontal edge has gradient pointing up/down, so the edge-normal
    angle is ±π/2 (vertical).  Hough normal = π/2, rho ≈ y_edge.
    """
    H, W = shape
    nms = np.zeros(shape, dtype=np.float64)
    angle = np.zeros(shape, dtype=np.float64)

    ys = []
    for i in range(n_edges):
        y = y_start + i * 3
        ys.append(y)
        nms[y, 10 : W - 10] = 100.0
        # Gradient-direction for horizontal edge: vertical (π/2)
        angle[y, 10 : W - 10] = np.pi / 2

    return nms, angle


def _synthetic_edges_vertical(
    shape=(50, 50),
    n_edges=4,
    x_start=20,
) -> tuple[np.ndarray, np.ndarray]:
    """Create (nms, angle) with vertical edges.

    A vertical edge has gradient pointing left/right, so the edge-normal
    angle is 0 or π.  Hough normal = 0, rho ≈ x_edge.
    """
    H, W = shape
    nms = np.zeros(shape, dtype=np.float64)
    angle = np.zeros(shape, dtype=np.float64)

    xs = []
    for i in range(n_edges):
        x = x_start + i * 3
        xs.append(x)
        nms[10 : H - 10, x] = 100.0
        angle[10 : H - 10, x] = 0.0

    return nms, angle


class TestHoughVotePeaks:
    def test_empty_input(self):
        """No edges → no peaks."""
        nms = np.zeros((20, 20), dtype=np.float64)
        angle = np.zeros((20, 20), dtype=np.float64)
        normals, rhos, scores = hough_vote_peaks(nms, angle)
        assert len(scores) == 0
        assert normals.shape == (0, 2)
        assert rhos.shape == (0,)

    def test_single_diagonal_edge(self):
        """A 45° diagonal edge should produce a peak near the diagonal line."""
        nms = np.zeros((30, 30), dtype=np.float64)
        angle = np.zeros_like(nms)
        # Draw a diagonal edge from (5,5) to (24,24)
        for i in range(5, 25):
            nms[i, i] = 100.0
            angle[i, i] = 3 * np.pi / 4  # gradient normal of a /-diagonal

        normals, rhos, scores = hough_vote_peaks(nms, angle, max_peaks=3)
        assert len(scores) >= 1
        # The highest peak should correspond to the diagonal line.
        best = np.argmax(scores)
        # The line x*cosθ + y*sinθ = ρ with θ = 3π/4 gives points like (10,10) → ρ ≈ 0.
        # Our coordinate origin is top-left, so a SE-slanting /-edge has ρ ≈ 0.
        assert abs(rhos[best]) < 25  # within the rho range of the image

    def test_horizontal_edges(self):
        """Horizontal edges → peaks with θ ≈ π/2, ρ matching edge y positions."""
        nms, angle = _synthetic_edges_horizontal(n_edges=3)
        # Use small NMS radii so that parallel lines 3 px apart aren't suppressed.
        normals, rhos, scores = hough_vote_peaks(
            nms, angle, theta_step_deg=1.0, nms_radius_theta=1, nms_radius_rho=1
        )

        # We should get at least 3 peaks (one per edge).
        assert len(scores) >= 3

        # Sort by rho ascending to match edge order.
        order = np.argsort(rhos)
        rhos_sorted = rhos[order]

        # Each horizontal edge at y=20, 23, 26: theta ≈ π/2, rho ≈ y.
        # At θ = π/2: rho = x*0 + y*1 = y.
        for i, expected_y in enumerate([20, 23, 26]):
            assert abs(rhos_sorted[i] - expected_y) < 3.0

    def test_vertical_edges(self):
        """Vertical edges → peaks with θ ≈ 0, ρ matching edge x positions."""
        nms, angle = _synthetic_edges_vertical(n_edges=3)
        # Use small NMS radii so that parallel lines 3 px apart aren't suppressed.
        normals, rhos, scores = hough_vote_peaks(
            nms, angle, theta_step_deg=1.0, nms_radius_theta=1, nms_radius_rho=1
        )

        assert len(scores) >= 3

        order = np.argsort(rhos)
        rhos_sorted = rhos[order]

        # At θ = 0: rho = x*1 + y*0 = x.
        for i, expected_x in enumerate([20, 23, 26]):
            assert abs(rhos_sorted[i] - expected_x) < 3.0

    def test_rho_non_negative(self):
        """All returned rho values should be ≥ 0 (canonicalised)."""
        nms, angle = _synthetic_edges_horizontal(n_edges=2)
        normals, rhos, scores = hough_vote_peaks(nms, angle)
        assert np.all(rhos >= 0)

    def test_normal_unit_length(self):
        """All returned normals should be unit vectors."""
        nms, angle = _synthetic_edges_horizontal(n_edges=2)
        normals, rhos, scores = hough_vote_peaks(nms, angle)
        assert len(normals) > 0
        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_array_almost_equal(lengths, 1.0)

    def test_scores_descending(self):
        """Peaks should be returned in descending score order."""
        nms, angle = _synthetic_edges_horizontal(n_edges=4)
        _, _, scores = hough_vote_peaks(nms, angle, max_peaks=10)
        assert len(scores) > 0
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_max_peaks_limit(self):
        """Should not return more than max_peaks."""
        nms, angle = _synthetic_edges_horizontal(n_edges=4)
        _, _, scores = hough_vote_peaks(nms, angle, max_peaks=2)
        assert len(scores) <= 2

    def test_threshold_rel_filters(self):
        """Peaks below the relative threshold are excluded."""
        nms, angle = _synthetic_edges_horizontal(n_edges=1)
        # With threshold_rel=1.5, nothing should pass.
        _, _, scores = hough_vote_peaks(nms, angle, threshold_rel=1.5, max_peaks=10)
        assert len(scores) == 0

    def test_single_pixel_edge(self):
        """Regression: a single edge pixel should not cause division errors."""
        nms = np.zeros((20, 20), dtype=np.float64)
        angle = np.zeros_like(nms)
        nms[10, 10] = 50.0
        angle[10, 10] = np.pi / 4
        normals, rhos, scores = hough_vote_peaks(nms, angle, max_peaks=5)
        assert len(scores) >= 1

    def test_nms_suppression_no_immediate_dup(self):
        """Peak NMS should suppress enough to avoid nearly identical peaks."""
        # Create a single thick-ish horizontal edge.
        H, W = 50, 50
        nms = np.zeros((H, W), dtype=np.float64)
        angle = np.zeros_like(nms)
        nms[25, 10:40] = 100.0
        angle[25, 10:40] = np.pi / 2

        normals, rhos, scores = hough_vote_peaks(
            nms, angle, theta_step_deg=2.0, nms_radius_theta=2, nms_radius_rho=4
        )
        # One strong horizontal edge should yield ~1 peak, not many.
        assert len(scores) >= 1
        # All rho values should be reasonably far apart from that one line.
        if len(scores) > 1:
            # Check that no two lines are within 5 px rho of each other AND
            # within 5° theta of each other.
            for i in range(len(rhos)):
                for j in range(i + 1, len(rhos)):
                    rho_dist = abs(rhos[i] - rhos[j])
                    # Dot product to get angular distance.
                    cos_dt = np.dot(normals[i], normals[j])
                    cos_dt = np.clip(cos_dt, -1.0, 1.0)
                    theta_dist = np.arccos(cos_dt)
                    # If rho is close, theta should be far apart; if theta is close,
                    # rho should be far apart.
                    if rho_dist < 5:
                        assert theta_dist > np.deg2rad(4), (
                            f"Duplicates: rho dist={rho_dist:.1f}, theta dist={np.rad2deg(theta_dist):.1f}°"
                        )


# ---------------------------------------------------------------------------
# refine_line
# ---------------------------------------------------------------------------


class TestRefineLine:
    @pytest.fixture
    def simple_horizontal_edge(self):
        """A clean horizontal edge at y=25 spanning x=[10, 40]."""
        H, W = 50, 50
        nms = np.zeros((H, W), dtype=np.float64)
        angle = np.zeros_like(nms)
        nms[25, 10:41] = 100.0
        return nms, angle

    def test_refine_horizontal(self, simple_horizontal_edge):
        """Refine should recover the exact horizontal line."""
        nms, angle = simple_horizontal_edge
        normal = np.array([0.0, 1.0], dtype=np.float64)  # approximate: θ ≈ π/2
        rho = 25.0  # approximate rho

        seg = refine_line(normal, rho, 100.0, nms, angle)

        # Normal should still point roughly upward (unit vector).
        assert np.linalg.norm(seg.normal) == pytest.approx(1.0)
        assert seg.normal[1] > 0.9  # dominant vertical component

        # rho should be close to the y position.
        assert seg.rho == pytest.approx(25.0, abs=2.0)

        # Endpoints should span the horizontal extent.
        xs = seg.endpoints[:, 0]
        assert xs.min() == pytest.approx(10.0, abs=3.0)
        assert xs.max() == pytest.approx(40.0, abs=3.0)

        # Y-coordinates of endpoints should be near 25.
        ys = seg.endpoints[:, 1]
        np.testing.assert_allclose(ys, 25.0, atol=2.0)

    def test_refine_passes_vote_score(self, simple_horizontal_edge):
        """The vote_score from the caller should appear unchanged in the segment."""
        nms, angle = simple_horizontal_edge
        normal = np.array([0.0, 1.0])
        seg = refine_line(normal, 25.0, 42.5, nms, angle)
        assert seg.vote_score == 42.5

    def test_refine_no_support(self):
        """When no edge pixels are near the candidate, endpoints are zero."""
        nms = np.zeros((20, 20), dtype=np.float64)
        angle = np.zeros_like(nms)
        nms[5, 5] = 1.0  # one pixel far away
        normal = np.array([1.0, 0.0])
        seg = refine_line(normal, 15.0, 10.0, nms, angle, distance_thresh=0.1)

        # The one pixel at (5,5) is 10 px from rho=15 → not within 0.1.
        # So support < 2 → degenerate.
        assert np.all(seg.endpoints == 0.0)

    def test_refine_single_support_pixel(self):
        """With exactly one support pixel → degenerate segment."""
        nms = np.zeros((20, 20), dtype=np.float64)
        angle = np.zeros_like(nms)
        nms[10, 10] = 100.0
        normal = np.array([1.0, 0.0])
        seg = refine_line(normal, 10.0, 10.0, nms, angle)
        # One pixel of support → < 2 → degenerate
        assert np.all(seg.endpoints == 0.0)

    def test_refine_vertical_line(self):
        """Refine a clean vertical edge."""
        H, W = 50, 50
        nms = np.zeros((H, W), dtype=np.float64)
        angle = np.zeros_like(nms)
        nms[10:41, 30] = 100.0  # vertical edge at x=30

        normal = np.array([1.0, 0.0], dtype=np.float64)  # θ ≈ 0
        rho = 30.0

        seg = refine_line(normal, rho, 100.0, nms, angle)

        assert seg.rho == pytest.approx(30.0, abs=2.0)
        assert seg.normal[0] > 0.9

        ys = seg.endpoints[:, 1]
        assert ys.min() == pytest.approx(10.0, abs=3.0)
        assert ys.max() == pytest.approx(40.0, abs=3.0)

        xs = seg.endpoints[:, 0]
        np.testing.assert_allclose(xs, 30.0, atol=2.0)

    def test_gap_bridging_joins_nearby_segments(self):
        """A small gap within tolerance should be bridged into one segment."""
        H, W = 30, 30
        nms = np.zeros((H, W), dtype=np.float64)
        angle = np.zeros_like(nms)
        # Two line fragments separated by 2 px gap (x=9 to x=11).
        nms[15, 5:10] = 100.0  # x=5..9
        nms[15, 11:17] = 100.0  # x=11..16
        # gap at x=10

        normal = np.array([0.0, 1.0], dtype=np.float64)
        rho = 15.0

        seg = refine_line(normal, rho, 100.0, nms, angle, gap_tolerance=2.5)

        # Should bridge the gap → endpoints span x=5 to x=16.
        xs = seg.endpoints[:, 0]
        assert xs.min() == pytest.approx(5.0, abs=2.0)
        assert xs.max() == pytest.approx(16.0, abs=2.0)

    def test_gap_tolerance_breaks_large_gaps(self):
        """A gap larger than tolerance should split segments."""
        H, W = 50, 50
        nms = np.zeros((H, W), dtype=np.float64)
        angle = np.zeros_like(nms)
        # Two fragments far apart.
        nms[15, 5:10] = 100.0
        nms[15, 30:35] = 100.0

        normal = np.array([0.0, 1.0], dtype=np.float64)
        rho = 15.0

        seg = refine_line(normal, rho, 100.0, nms, angle, gap_tolerance=2.0)

        # Should pick the longer of the two fragments.
        # Both have length ~5, so either is fine. But endpoints must span only ONE fragment.
        xs = seg.endpoints[:, 0]
        span = xs.max() - xs.min()
        # Each fragment is ~5 px wide; a bridged version would be ~25 px.
        assert span < 10.0

    def test_weighted_tls_strong_edges_dominate(self):
        """Strong edge pixels should pull the line more than weak ones."""
        H, W = 30, 30
        nms = np.zeros((H, W), dtype=np.float64)
        angle = np.zeros_like(nms)
        # Both within distance_thresh (2.0) of the candidate line at rho=12.5.
        nms[12, 5:15] = 10.0  # weak, at y=12, rho≈12 → within 0.5 px
        nms[14, 5:15] = 100.0  # strong, at y=14, rho≈14 → within 1.5 px

        normal = np.array([0.0, 1.0], dtype=np.float64)
        rho = 12.5  # midway between 12 and 14

        seg = refine_line(normal, rho, 100.0, nms, angle, distance_thresh=2.0)

        # Weighted TLS should pull the line closer to y=14 (strong edge).
        assert seg.rho > 12.5


# ---------------------------------------------------------------------------
# Phase III — Synthetic isolation tests (plan-006)
# ---------------------------------------------------------------------------


class TestRefineLineRealistic:
    """Synthetic isolation tests for each documented failure mode.

    Each test builds a minimal synthetic (nms, angle) pair that isolates one
    failure mode with realistic pixel patterns informed by the diagnostic
    debug output from debug_hough_failures.py.
    """

    # ---- A: gap_tolerance too small for real NMS gaps ----------------------

    def test_isolation_A_gap_tolerance_insufficient(self):
        """Failure A: gap_tolerance=2.0 can't bridge real 4-7 px NMS gaps.

        Diagnostic output showed finder boundaries with 4-7 px NMS gaps
        (gradient leakage → intermittent dropouts). With gap_tolerance=2.0
        the longest contiguous run covers only one fragment (~8-12 px),
        not the full finder boundary (~30-40 px).
        """
        H, W = 20, 80
        nms = np.zeros((H, W), dtype=np.float64)
        angle = np.zeros_like(nms)

        # 4 clusters mimicking real NMS output from debug:
        #   cluster 1: x=5..13 (9 px), gap 4 px, cluster 2: x=17..25 (9 px),
        #   gap 7 px, cluster 3: x=32..38 (7 px), gap 3 px, cluster 4: x=41..45 (5 px)
        for x in range(5, 14):
            nms[10, x] = 200.0
            angle[10, x] = np.pi / 2
        for x in range(17, 26):
            nms[10, x] = 200.0
            angle[10, x] = np.pi / 2
        for x in range(32, 39):
            nms[10, x] = 200.0
            angle[10, x] = np.pi / 2
        for x in range(41, 46):
            nms[10, x] = 200.0
            angle[10, x] = np.pi / 2

        normal = np.array([0.0, 1.0], dtype=np.float64)
        rho = 10.0

        seg = refine_line(
            normal, rho, 100.0, nms, angle, gap_tolerance=2.0, distance_thresh=1.5
        )

        assert not np.all(seg.endpoints == 0), "Should find SOME segment"

        xs = seg.endpoints[:, 0]
        span = abs(float(xs[1] - xs[0]))
        # Full visible span is 40 px (5→45).  With gap_tolerance=2.0 the
        # 4px and 7px gaps break the run.  The longest cluster is ~9 px.
        # THIS IS THE BUG: we should bridge at least the 3-4px gaps.
        assert span < 20.0, (
            f"BUG CONFIRMED: gap_tolerance=2.0 bridges 4+ px gaps — span={span:.1f} < 20"
        )

    # ---- B: sparse coincidental alignment looks like a line --------------

    def test_isolation_B_sparse_noise_creates_phantom(self):
        """Failure B: 6 sparse noise pixels at a diagonal + strong orthogonal
        edge nearby.  TLS refinement drags the diagonal normal toward the
        nearby strong edge, creating a spurious segment.

        This replicates the Cluster 3 phantom pattern: a peak at 152°
        (normal of a strong internal QR module edge) whose TLS-refined
        segment picks up support from the high-density NMS region.
        """
        H, W = 60, 60
        nms = np.zeros((H, W), dtype=np.float64)
        angle = np.zeros_like(nms)

        # Strong horizontal edge A (real finder structure) at y=20, x=10..30
        for x in range(10, 31):
            nms[20, x] = 200.0
            angle[20, x] = np.pi / 2

        # Strong vertical edge B (real internal structure) at x=45, y=10..40
        for y in range(10, 41):
            nms[y, 45] = 200.0
            angle[y, 45] = 0.0

        # 6 sparse diagonal "noise" pixels at ~26.6° slope — coincidentally
        # aligned enough to produce a Hough peak with normal ≈ 116°
        diagonal_pts = [(5, 50), (10, 42), (15, 34), (20, 26), (25, 18), (30, 10)]
        for dx_idx, dy_idx in diagonal_pts:
            nms[dy_idx, dx_idx] = 60.0
            angle[dy_idx, dx_idx] = 3 * np.pi / 4  # gradient normal of /-diagonal

        normals, rhos, scores = hough_vote_peaks(
            nms, angle, theta_step_deg=2.0, max_peaks=15
        )

        # Should have peaks for horizontal (θ≈90°) and diagonal (θ≈116°)
        diag_idx = None
        for i, n in enumerate(normals):
            ang = np.rad2deg(np.arctan2(n[1], n[0]))
            if 110 <= ang <= 125:
                diag_idx = i
                break

        assert diag_idx is not None, (
            "Should detect a diagonal peak from 6 sparse pixels"
        )

        # Now refine. With only 6 support pixels, TLS is unstable.
        # The refined normal drifts toward the strong horizontal edge B.
        seg = refine_line(
            normals[diag_idx],
            float(rhos[diag_idx]),
            scores[diag_idx],
            nms,
            angle,
            gap_tolerance=5.0,
            distance_thresh=3.0,
        )

        assert not np.all(seg.endpoints == 0), (
            "6 sparse coincidental pixels produce a segment — this IS the phantom bug"
        )

        # The segment endpoints should have a non-trivial span
        span = float(np.linalg.norm(seg.endpoints[1] - seg.endpoints[0]))
        assert span > 5.0, f"Phantom span={span:.1f} — too small to matter"

    # ---- C: TLS direction drift captures adjacent parallel edge -----------

    def test_isolation_C_tls_drift_bridges_parallel_edges(self):
        """Failure C: weighted TLS refines the direction ~1° away from the
        Hough peak direction, causing the segment to capture support from a
        parallel nearby edge.

        Two horizontal edges at rho=25 (target) and rho=28 (pollution).
        The TLS on the 25-pixel edge should not bridge across the 3-px
        perpendicular gap to the 28-pixel edge. But with distance_thresh=2.0
        and normal drift, it does.
        """
        H, W = 50, 80
        nms = np.zeros((H, W), dtype=np.float64)
        angle = np.zeros_like(nms)

        # Edge A: horizontal at row 25, x=10..30 (strong, continuous)
        for x in range(10, 31):
            nms[25, x] = 200.0
            angle[25, x] = np.pi / 2

        # Edge B: horizontal at row 28, x=50..70 (strong, continuous)
        # 3 px away perpendicularly — within distance_thresh depending on normal
        for x in range(50, 71):
            nms[28, x] = 200.0
            angle[28, x] = np.pi / 2

        normal = np.array([0.0, 1.0], dtype=np.float64)
        rho = 25.0

        seg = refine_line(
            normal,
            rho,
            100.0,
            nms,
            angle,
            gap_tolerance=2.0,
            distance_thresh=2.0,
        )

        assert not np.all(seg.endpoints == 0), "Should produce a segment"

        xs = seg.endpoints[:, 0]
        span = abs(float(xs[1] - xs[0]))
        # Edge A alone spans x=10→30 = 20 px (plus TLS can slightly extend).
        # With pollution from edge B (x=50→70), the span could jump to 60 px.
        assert span <= 30.0, (
            f"BUG CONFIRMED: TLS bridges 3 px rho gap to parallel edge B — span={span:.1f} > 30"
        )

    # ---- D: Hough quantization pushes rho out of matching gate ------------

    def test_isolation_D_hough_quantization_misses_peak(self):
        """Failure D: with 2° theta bins, the quantized theta produces
        a rho that can be 10-15 px off the true rho, causing the peak
        to fall outside a 5 px rho matching gate even though the angle
        matches perfectly.

        Real example from debug: GT normal θ=145.7°, ρ=24.3.
        Hough bin at θ=146°, quantised rho computed with cos(146°) instead
        of cos(145.7°) → rho error ~11 px.
        """
        H, W = 60, 60
        nms = np.zeros((H, W), dtype=np.float64)
        angle = np.zeros_like(nms)

        # A perfectly diagonal edge (45°) at y=2*x (so normal at -45° → mod π = 135°)
        # This should produce a Hough peak at θ_bin ≈ 134° or 136° (2° steps)
        # The rho from the quantized theta will be off by several px.
        true_theta = 3 * np.pi / 4  # 135° normal for a /-diagonal
        for i in range(5, 25):
            x, y = i, 2 * i + 10
            if 0 <= x < W and 0 <= y < H:
                nms[y, x] = 200.0
                angle[y, x] = true_theta

        normals, rhos, scores = hough_vote_peaks(
            nms, angle, theta_step_deg=2.0, max_peaks=10
        )

        # With 2° steps, 135° falls exactly on a bin boundary. The Hough
        # voting quantizes to either 134° or 136°, shifting rho.
        assert len(normals) >= 1, "Should find at least one peak"

        # Compute the true rho at the exact diagonal
        true_normal = np.array([-np.sqrt(2) / 2, np.sqrt(2) / 2], dtype=np.float64)
        example_point = np.array([10.0, 30.0])  # point on y=2x+10 at x=10
        true_rho = float(true_normal @ example_point)

        # Check if any Hough peak is within 5° AND 5 px
        found = False
        for i in range(len(normals)):
            dot = np.clip(np.abs(np.dot(normals[i], true_normal)), -1.0, 1.0)
            ang_dist = float(np.rad2deg(np.arccos(dot)))
            rho_dist = abs(float(rhos[i]) - abs(true_rho))
            if ang_dist < 5.0 and rho_dist < 5.0:
                found = True
                break

        if not found:
            # The bug: quantized theta creates rho error that exceeds 5 px gate
            best = min(
                range(len(normals)),
                key=lambda i: (
                    abs(float(rhos[i]) - abs(true_rho))
                    + float(
                        np.rad2deg(
                            np.arccos(
                                np.clip(
                                    np.abs(np.dot(normals[i], true_normal)), -1.0, 1.0
                                )
                            )
                        )
                    )
                ),
            )
            dot = np.clip(np.abs(np.dot(normals[best], true_normal)), -1.0, 1.0)
            ang_dist = float(np.rad2deg(np.arccos(dot)))
            rho_dist = abs(float(rhos[best]) - abs(true_rho))
            pytest.fail(
                f"QUANTIZATION BUG: best peak at {ang_dist:.1f}°/"
                f"{rho_dist:.1f}px — theta binning pushes rho outside 5°+5px gate"
            )

    # ---- D2: degenerate when only 3-4 edge pixels exist -------------------

    def test_isolation_D2_few_pixels_become_degenerate(self):
        """Failure D (degeneracy): when a real edge has only 3-4 NMS-surviving
        pixels, the weighted-TLS SVD can still compute a direction, but the
        contiguous-run logic produces zero-length endpoints.

        Real scenario: a short finder boundary edge in a small ROI where
        most pixels were suppressed by NMS, leaving only sparse edge pixels.
        """
        H, W = 30, 30
        nms = np.zeros((H, W), dtype=np.float64)
        angle = np.zeros_like(nms)

        # 3 edge pixels at x=10, 14, 18 on row 15 (gap 4 px between each)
        for x in (10, 14, 18):
            nms[15, x] = 200.0
            angle[15, x] = np.pi / 2

        normal = np.array([0.0, 1.0], dtype=np.float64)
        rho = 15.0

        seg = refine_line(
            normal, rho, 100.0, nms, angle, gap_tolerance=2.0, distance_thresh=2.0
        )

        if np.all(seg.endpoints == 0):
            # The bug manifests: 3 pixels with 4px gaps → each gap > 2.0
            # breaks the run → longest run is 1 pixel → degenerate.
            pytest.fail(
                "DEGENERACY: 3 edge pixels 4px apart → refine_line returns zero endpoints "
                "(gap_tolerance=2.0 can't bridge 4px gaps)"
            )
        else:
            # If it somehow works, check span is reasonable
            xs = seg.endpoints[:, 0]
            span = abs(float(xs[1] - xs[0]))
            assert span >= 5.0, f"Span too small: {span:.1f} px"
