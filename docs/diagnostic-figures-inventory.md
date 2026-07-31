# Diagnostic Figure Inventory

## Context

- Ticket: https://github.com/RikVoorhaar/qr-reader/issues/10
- Scope: local repository resources only; no external sources or blog prose.
- Branch: `research/diagnostic-figures`
- Inventory date: 2026-07-31

## Availability Legend

- **Runnable generator**: current code can produce the figure, but it is
  interactive or writes to an ignored location rather than providing a
  committed asset.
- **Local output**: an output file exists in the checkout, normally as an
  ignored diagnostic JSON file. It is evidence for stage behavior, not a
  figure.
- **Historical generator**: the generator or report exists in local Git
  history but was deleted from the working tree. It must not be described as
  an existing current asset.
- **New figure justified**: the current repository does not provide a compact,
  current, directly reusable visual for the section.

## Repository Asset Audit

| Resource | Finding |
|---|---|
| Notebooks (`*.ipynb`) | None found. Notebook-style scripts use `# %%` cells instead. |
| Tracked QR diagnostic images/PDFs/SVGs | None found through the tracked-file inventory. The images under `blog/static/blog/` are assets for unrelated posts. |
| `diagnose_images/` | No generated images currently present. The directory is ignored (`.gitignore:216`). |
| `data/` | No dataset files currently present in the checkout. The directory is ignored (`.gitignore:210`). Several generators require `data/images/train/*.jpg` or benchmark images. |
| Current plot policy | Current notebook-style diagnostics call `plt.show()`; `diagnose_images.py` is the exception and saves PNGs to the ignored `diagnose_images/` directory. |
| Ignored JSON outputs | `diagnose_composited.json`, `benchmark_current.json`, `benchmark_ray.json`, and `benchmark_captured.json` exist locally and are ignored (`.gitignore:217-220`). |

## Section Inventory

### Finder-pattern structure

| Candidate | What it explains | Status and assessment |
|---|---|---|
| `src/qr_reader/scripts/verify_finder_segments.py:1-203,267-336` | A generated v12 image, 36 ground-truth finder edge segments (12 per finder), and per-ROI overlays. The full-image plot color-codes TL/TR/BL finders; ROI plots show which GT segments intersect each cluster. | **Runnable generator.** It is the strongest current structural geometry diagnostic, uses a synthetic gradient background, and does not require the absent HomeObjects images. It shows boundary geometry rather than a labeled 7x7 finder-pattern explanation. |
| `src/qr_reader/detector/ray_fit.py:51-88` | The current radial soft template: dark 3-module center, white ring, dark ring, then quiet zone. | **Source only.** The model is explicit but has no standalone structure figure. |
| `src/qr_reader/scripts/ray-profile.py:319-359,625-691,1095-1141` | Finder ROI with 36 half-rays, ray-profile heatmap, per-ray fitted module pitch, and actual-versus-theoretical profiles. | **Runnable generator.** Useful for showing observed finder structure and why radial fitting works; it is a fitting diagnostic, not a clean introductory structure diagram. |
| `src/qr_reader/qr_gen.py:192-224` | Canonical finder-pattern corner geometry and the 7x7-module definition used by synthetic ground truth. | **Source only.** |
| `src/qr_reader/scripts/verify_finder_segments.py:186-203` | Existing full-image structural overlay. | **Reuse.** Prefer this over inventing a new full-image geometry plot. A small new clean finder crop with 1:1:3:1:1 labels is justified only if the blog needs the QR structure itself rather than detector geometry. |

### Otsu and run-length candidates

| Candidate | What it explains | Status and assessment |
|---|---|---|
| `src/qr_reader/qr_gen.py:107-117` | Otsu thresholding and the boolean convention (`True = white`). | **Source only.** |
| `src/qr_reader/scripts/full-pipeline-current.py:111-148` | Side-by-side grayscale/Otsu image and all 2-D validated candidates drawn as five colored horizontal segments. | **Runnable generator, but incomplete later.** This is the best current candidate for the Otsu and candidate stage. It requires `data/images/train/*.jpg` and ends at the clustering stage (`:185-187` says finder fitting was deleted). |
| `src/qr_reader/scripts/diagnose_visual.py:61-65,155-161` | Otsu, 2-D candidate positions, and a full-image overlay. | **Runnable generator.** Requires a HomeObjects background and is interactive. |
| `src/qr_reader/detector/alignment.py:7-88` | The actual run-length encoding, 1:1:3:1:1 score, horizontal scan, vertical scan, and cross-validation. | **Source only.** |
| `src/qr_reader/scripts/verify_finder_segments.py:205-220` | Candidate count and cluster list after running the same binarize/scan stage. | **Runnable generator.** Output is mostly textual; its later ROI plots are more useful than the candidate stage itself. |
| `diagnose_composited.json` | 80 composited cases: 2,352 candidate rows and 244 clusters in aggregate; errors include 4 `No alignment patterns found` cases. | **Local output.** No candidate pixels or images are stored. The JSON supports a failure-count callout, not a visual explanation. |
| `src/qr_reader/scripts/diagnose_images.py:109-151` | Batch annotated PNG with Otsu-derived candidates overlaid on the input, plus GT/finder overlays. | **Runnable generator.** The PNG output is currently absent and ignored. |
| `src/qr_reader/scripts/full-pipeline.py` at historical parent `f5451e8^` | Historical plots for horizontal candidates, 2-D candidates, and colored run-length segments. | **Historical generator.** The current equivalent is `full-pipeline-current.py`; do not rely on the deleted script as an existing asset. |

**Figure decision:** reuse the current grayscale/Otsu plus candidate overlay. A
new focused 1-D inset showing the five run lengths and the vertical
cross-validation is justified if the section must explain the ratio test; no
such inset exists in the current generators.

### Clustering and confirmation

| Candidate | What it explains | Status and assessment |
|---|---|---|
| `src/qr_reader/scripts/full-pipeline-current.py:150-183` | Candidate clusters as colored bounding rectangles, centers, widths, and heights. | **Runnable generator.** Direct visual for the cluster stage, but it requires absent background data. |
| `src/qr_reader/scripts/diagnose_visual.py:121-161` | Cluster bounding boxes, candidate points, fitted finder quadrilaterals, GT QR outline, and detected QR outline when detection succeeds. | **Runnable generator.** Best current end-to-end confirmation overlay; it is interactive and tuned to failure cases. |
| `src/qr_reader/scripts/diagnose_images.py:116-144` | Same confirmation overlay rendered to a PNG for selected success/failure cases. | **Runnable generator; no output present.** The script selects representative cases from `diagnose_composited.json` when invoked with `--all`. |
| `src/qr_reader/detector/clustering.py:32-73,95-140` | The overlap predicates and weighted merge used to turn candidates into `CandidateCluster` objects. | **Source only.** |
| `src/qr_reader/scripts/ray-profile.py:571-789` | Per-ray pitch fits and the estimated 3.5-module boundary points/bounding box. | **Runnable generator.** This is the strongest current confirmation diagnostic before corner extraction. |
| `benchmark_captured.json` | On 100 captured images, 65 reach at least three candidates and 64 reach at least three fitted finders. | **Local output.** The source defines these funnel stages at `benchmark_captured.py:157-166`; the JSON has no corresponding images. |

**Figure decision:** the existing cluster overlay plus one ray-profile/ROI
figure is sufficient. A new figure is justified only to make the merge
operation explicit by showing raw candidates and the merged cluster in one
zoomed panel; no current plot shows both states with a merge relationship.

### Corner fitting

| Candidate | What it explains | Status and assessment |
|---|---|---|
| `src/qr_reader/scripts/ray-profile.py:1271-1379` | Initial and jointly refined four lines, colored boundary points, ray transition markers, GT quadrilateral, and the refined finder outline. | **Runnable generator.** This is the best current corner-fitting figure. It requires the absent background dataset and may emit one plot per cluster. |
| `src/qr_reader/scripts/ray-profile.py:1382-1477,1480-1606` | Individual ray profiles versus the joint template, and actual-versus-template heatmaps. | **Runnable generator.** Useful supporting diagnostics, but visually dense for a blog figure. |
| `src/qr_reader/scripts/diagnose_visual.py:67-86,138-145,165-194` | Current production ray fit and fitted finder polygons, including ROI cutouts. | **Runnable generator.** Simpler than `ray-profile.py`, but it does not expose the fitting evidence or initial/refined distinction. |
| `src/qr_reader/scripts/debug_gt_vs_refined.py:216-268` | Numeric initial-versus-refined side errors against GT. | **Source/generator, text only.** No plot is produced. |
| `src/qr_reader/detector/ray_fit.py:154-527` | Current radial sampling, module-pitch fitting, boundary-point construction, edge clustering, and result validity. | **Source only.** |
| `src/qr_reader/scripts/full-pipeline-current.py:185-187` | Explicitly records that the old per-cluster finder fitting was deleted and not reimplemented there. | **Important gap.** Do not present `full-pipeline-current.py` as a complete current corner-fitting figure. |

**Figure decision:** reuse the joint-refinement ROI plot if a generated image
can be selected. A new compact four-panel figure is justified for the blog:
ROI with initial/refined/GT quadrilaterals, one representative radial profile
with the soft template, boundary points grouped into four edges, and the final
corner error. No current saved asset provides that compact progression.

### Rejected flood-fill and Hough approaches

| Candidate | What it explains | Status and assessment |
|---|---|---|
| `src/qr_reader/detector/region.py:37-65,67-103` | Current wave-front flood fill, 8-connected boundary extraction, and connected components. | **Current source, not current production path.** No current plot generator visualizes these masks. |
| Historical `src/qr_reader/scripts/full-pipeline.py` at `f5451e8^` | Queue fill, wave-front fill, boundary mask, connected components, radial-distance/angular-NMS corner plot, all-corners overlay, finder associations, triplets, landmarks, version constraints, homography, sampled bits, and decode result. | **Historical generator.** It is the most complete source for the abandoned flood-fill visualization, but it depends on the old pipeline and data layout and is deleted from the working tree. |
| Historical `docs/hough-failure-analysis.md` and `docs/hough-pipeline-catalogue.md` at `f5451e8^` | Hough failure modes A-D, accumulator/support diagnostics, and experiment results. | **Historical reports, not assets.** They record numeric evidence, including v12-default totals (A=2, B=5, C=4, D=2) and HoughLinesP/Canny comparisons, but no committed image output. |
| Historical `src/qr_reader/scripts/full-pipeline-canny.py` at `f5451e8^` | Three Hough figure families: edge extraction, accumulator heatmap with GT markers, and detected/GT line overlays. | **Historical generator.** It is a strong source for a rejected-Hough explanation but is not part of the current tree. |
| `src/qr_reader/scripts/debug_find_all_associations.py:169-220,224-375,486-620` | Reproduction of the old region-fill-to-association path plus verbose pair rejection diagnostics and experimental association strategies. | **Current debug generator, text only.** It does not produce plots and uses deprecated association functions. |
| Commit history | `f5451e8` removed all Hough implementation, tests, and scripts; `a984886` removed the associated reports and older plot scripts. | **Evidence of rejection.** The current detector imports `ray_fit`, not Hough, at `src/qr_reader/detector/detector.py:20-35`. |

**Figure decision:** a new figure is justified if this planned section is kept.
The repository has no current flood-fill or Hough image asset. The smallest
useful replacement would be one reconstructed historical comparison: binary
ROI -> flood-filled region/boundary -> angular-NMS corners on the left, and a
single Hough accumulator/line failure on the right, clearly labeled as
historical/rejected. Do not imply that either path is the current detector.

### Triplets, version, and homography

| Candidate | What it explains | Status and assessment |
|---|---|---|
| `src/qr_reader/detector/finder_pattern.py:274-444` | Current triplet discovery: module-size compatibility, center distance, axis alignment, non-connected neighbor rule, right-angle check, and role assignment. | **Source only.** No current triplet plot. |
| `src/qr_reader/detector/detector.py:227-315` | Current triplet selection, legal-version search (`N = 21..177`), DLT + LM homography per candidate, timing-pattern score, and selected version. | **Source only.** The production path has no score-curve or landmark visualization. |
| `src/qr_reader/scripts/diagnose_visual.py:106-117,147-153` | Counts triplets and, if full detection succeeds, overlays the projected QR boundary. | **Runnable generator.** It confirms outcome but does not show how the triplet or version was chosen. |
| `src/qr_reader/scripts/debug_find_all_associations.py:387-440` | Compact version/seed association sweep with production and experimental association counts. | **Runnable generator, text only.** It targets the deprecated association route, not the current triplet routine. |
| Historical `src/qr_reader/scripts/full-pipeline.py` at `f5451e8^` | Ordered finder squares A-F, colinear quadruples, kept/rejected version constraints, top-five version scores, RANSAC inliers, LM homography, and projected QR corners. | **Historical generator.** It is not safe to reuse as a current pipeline figure without updating its algorithm labels. |
| `benchmark_captured.json` | 49/100 captured images reach a valid triplet and homography/version stage; 40/100 are decodable and have the expected payload. | **Local output.** This is a useful quantitative funnel companion, not a geometry figure. |
| `diagnose_composited.json` | 48/80 cases end with `No finder-pattern triplet found`; 28/80 complete detection and decode. | **Local output with a schema caveat.** Its `n_triplets` field is zero throughout because the diagnostic script does not copy the full detector's triplet count into the result object; use the error field and success count, not that field, for the funnel. |

**Figure decision:** a new current figure is strongly justified. The minimum
useful figure is a three-finder overlay with the chosen L-shaped triplet,
named landmark correspondences, projected QR boundary, and a small version
candidate score curve. The historical figure family covers this concept but
uses the removed cross-ratio/association route, while the current production
route searches homographies and scores timing-pattern alternation.

### Decoder handoff

| Candidate | What it explains | Status and assessment |
|---|---|---|
| `src/qr_reader/detector/sample.py:143-183` | Homography-based 3x3 supersampling, adaptive thresholding from known finder cells, and the boolean matrix handed to the decoder. | **Source only.** No current sampled-matrix plot. |
| `src/qr_reader/decoder/decoder.py:28-101` | Format info -> version cross-check -> unmask/zigzag codewords -> de-interleave -> Reed-Solomon -> bitstream text. | **Source only.** The decoder stages are algorithmic and have no plot generation. |
| Historical `src/qr_reader/scripts/full-pipeline.py` at `f5451e8^` | Display of the sampled QR matrix followed by a decode success/failure assertion. | **Historical generator.** It is the only located matrix visualization, and it belongs to the removed pipeline. |
| `benchmark_current.json` | 180 synthetic cases: 78 detection successes and 52 successful decodes. | **Local output.** Quantitative evidence only. |
| `benchmark_ray.json` | 180 synthetic cases from the ray benchmark: 62 detection successes and 60 successful decodes. | **Local output.** Quantitative evidence only; no matrices or failure images. |
| `benchmark_captured.json` | 40/100 captured images decode to the expected `rikvoorhaar.com` payload. | **Local output.** Quantitative handoff outcome only. |

**Figure decision:** a new current handoff figure is justified. Use one
rectified boolean module matrix with the detector-to-decoder boundary marked,
plus a small annotation naming `format_info`, `version_info` (when applicable),
codeword extraction, Reed-Solomon, and bitstream decode. The repository has no
current artifact for this handoff.

## Recommended Figure Set

| Priority | Figure | Basis |
|---:|---|---|
| 1 | Otsu image + colored 2-D run-length candidates + cluster boxes | Current generators already cover the first two detection stages (`full-pipeline-current.py:111-183`). |
| 2 | Finder ROI with radial profiles, soft 1:1:3:1:1 template, fitted four sides, and GT comparison | Current `ray-profile.py` has all ingredients but spreads them across dense per-cluster plots. |
| 3 | Current triplet, selected version evidence, homography, and projected QR boundary | No current plot; historical version/landmark plot is algorithmically stale. |
| 4 | Rectified sampled bit matrix handed to the decoder | No current plot; only historical generator exists. |
| 5 | Rejected flood-fill/Hough comparison | No current asset; add only if the planned section needs the rejection rationale. |
| 6 | Clean finder-pattern structure crop with labeled rings | Optional. `verify_finder_segments.py` covers GT geometry, while the current template source covers the structure but neither is a compact teaching figure. |

## Bottom Line

- The repository can currently generate useful visuals for Otsu/candidates,
  clustering, ray profiles, and fitted finder corners, but the generators are
  interactive or depend on ignored datasets.
- The only durable local diagnostic outputs are ignored JSON summaries; there
  are no current committed QR diagnostic figures and no generated
  `diagnose_images/` output.
- Flood-fill, Hough, landmarks/version-constraint, and sampled-matrix visuals
  are recoverable only from deleted historical scripts. They are evidence of
  prior work, not current assets.
- The most justified new plots are the current triplet/version/homography
  explanation and the detector-to-decoder sampled-matrix handoff. A compact
  corner-fitting plot is also justified if the blog needs more than the
  existing dense ray-profile diagnostics.
