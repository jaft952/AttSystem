# CBIR Method Performance Report

Evaluation of both CBIR preprocessing pipelines on a shared held-out test split.
All results are produced by `ml/evaluation.ipynb`.

---

## Dataset & Split

| Parameter | Value |
|---|---|
| Identities | 21 |
| Gallery ratio | 80 % |
| Test ratio | 20 % |
| Split seed | 42 |
| Total gallery images read | 14 630 |

---

## Preprocessing Pipelines

Both methods start from a grayscale image.
A Haar cascade detects the largest face ROI (padded 20 %) before the method-specific steps below are applied.

| Step | Method 1 | Method 2 |
|---|---|---|
| Contrast | CLAHE (clip = 2.0, grid = 8 × 8) | Global histogram equalisation |
| Smoothing | Gaussian blur (3 × 3) | Bilateral filter (d = 7, σ_color = 50, σ_space = 50) |
| Sharpening | — | Unsharp mask (weight 1.35 / −0.35, σ = 1.2) |
| Resize | 128 × 128 (cubic) | 128 × 128 (cubic) |

Embeddings are extracted with `face_recognition` (128-dim dlib ResNet), L2-normalised, and matched by cosine distance.

---

## Gallery Index Statistics

| Metric | Method 1 | Method 2 |
|---|---|---|
| Gallery images attempted | 14 630 | 14 630 |
| Gallery vectors indexed | 11 653 | 11 341 |
| Embedding dimension | 128 | 128 |

---

## Evaluation Results

Thresholds were identical for both methods during evaluation.

| Parameter | Value |
|---|---|
| Similarity threshold | 0.60 |
| Strict unknown threshold | 0.72 |
| Min margin | 0.035 |

| Metric | Method 1 | Method 2 |
|---|---|---|
| Evaluated samples | 2 913 | 2 834 |
| Known samples | 2 913 | 2 834 |
| Unknown samples | 0 | 0 |
| **Overall accuracy** | **60.38 %** | **87.33 %** |
| Known accuracy | 60.38 % | 87.33 % |
| Unknown reject rate | N/A | N/A |

**Winner: Method 2 (+26.95 pp over Method 1)**

---

## Per-Identity Gallery Yield

### Method 1

| Identity | Indexed | Attempted |
|---|---|---|
| benjamin | 469 | 864 |
| chern_tak | 805 | 806 |
| chillien | 112 | 259 |
| daniel | 680 | 864 |
| dylan | 312 | 605 |
| han_soon | 384 | 576 |
| harry | 281 | 288 |
| isaac | 33 | 288 |
| jing_ang | 650 | 864 |
| jun_wei | 715 | 778 |
| kang_kai | 839 | 864 |
| marion | 600 | 806 |
| ms_nurul | 297 | 317 |
| qi_xuan | 838 | 864 |
| shuang_quan | 1 161 | 1 296 |
| wee_xuan | 636 | 691 |
| xiang_yue | 841 | 864 |
| xu_sheng | 820 | 835 |
| yoke_hong | 407 | 922 |
| yong_kang | 652 | 691 |
| zi_herng | 121 | 288 |
| **Total** | **11 653** | **14 630** |

### Method 2

| Identity | Indexed | Attempted |
|---|---|---|
| benjamin | 419 | 864 |
| chern_tak | 805 | 806 |
| chillien | 85 | 259 |
| daniel | 686 | 864 |
| dylan | 304 | 605 |
| han_soon | 302 | 576 |
| harry | 278 | 288 |
| isaac | 38 | 288 |
| jing_ang | 632 | 864 |
| jun_wei | 713 | 778 |
| kang_kai | 839 | 864 |
| marion | 582 | 806 |
| ms_nurul | 298 | 317 |
| qi_xuan | 838 | 864 |
| shuang_quan | 1 089 | 1 296 |
| wee_xuan | 636 | 691 |
| xiang_yue | 841 | 864 |
| xu_sheng | 819 | 835 |
| yoke_hong | 392 | 922 |
| yong_kang | 634 | 691 |
| zi_herng | 111 | 288 |
| **Total** | **11 341** | **14 630** |

---

## Observations

- **Method 2 outperforms Method 1 by ~27 percentage points** on overall known-identity accuracy.
- Identities with low yield (e.g. *isaac*, *zi_herng*, *chillien*) are likely to have poorer per-person accuracy due to fewer usable gallery embeddings; this is consistent across both methods.
- Both methods use the same embedding model and decision thresholds, so the accuracy gap is attributable entirely to the preprocessing pipeline.
- Method 2's combination of global histogram equalisation, bilateral denoising, and unsharp masking produces embeddings that better separate identities under this dataset's lighting conditions.
- No "unknown-only" identities were present in this evaluation run, so the unknown reject rate is not applicable.

---

## Artefact Paths

| Artefact | Path |
|---|---|
| Method 1 index | `models/cbir_method1_index.npz` |
| Method 1 meta | `models/cbir_method1_meta.json` |
| Method 2 index | `models/cbir_method2_index.npz` |
| Method 2 meta | `models/cbir_method2_meta.json` |
| Shared eval split | `models/cbir_eval_split.json` |
| Evaluation notebook | `ml/evaluation.ipynb` |
