# ADM Evaluation Pipeline

Scripts to evaluate three image-filtering baselines (Random, Generative Uncertainty, Realism) against the ADM model on ImageNet-128.

All paths are configured in `vars.sh` and exported to each sub-script by `gpu_job_launcher.sh`.

---

## Pipeline Overview

```
1_dataset.sh          (one-time)  package ImageNet val images into a zip
2_fid_ref_stats.sh    (one-time)  compute FID reference stats + real-image P&R features
3_all_features.sh                 extract Inception features for every generated image
3_realism_eval.sh                 compute per-image realism score from those features
4_baselines.sh  N                 evaluate Random / GU / Realism at budget N
```

Steps 1 and 2 are one-time setup steps. Steps 3 → 4 are run each time the generated image set changes.

---

## Scripts

### `vars.sh`
Exports three path variables used by all other scripts:
- `DATA_PATH` – ImageNet validation images (source)
- `ROOT_PATH` – root output directory (FID refs, P&R refs)
- `EXP_PATH` – path to the specific experiment (generated images, features, scores)

### `1_dataset.sh`
Packages the ImageNet validation images into a 128×128 center-cropped zip that `fid.py ref` expects.

### `2_fid_ref_stats.sh`
Runs `fid.py ref` to pre-compute:
- FID reference statistics (`fid-refs/imagenet-128x128.npz`)
- Real-image Inception features (`precision-recall-refs/image_net_val_128_fid_features_.pt`)

Both files are reused across all experiments and budget levels.

### `3_all_features.sh`
Runs `fid.py calc --num 0` to extract Inception-v3 features **for every** generated image
(passing `--num 0` disables the dataset-size cap in `fid.py`).

Output: `$EXP_PATH/0/fid_features_all.pt`

This step **must** run before `3_realism_eval.sh`.

### `3_realism_eval.sh`
Runs `precision_recall_torch.py --realism` on `fid_features_all.pt` to produce a
**per-image realism score**: how well each generated image sits inside the real-data manifold.

Output: `$EXP_PATH/0/realism.npy`  (shape: `[total_images]`)

**Known limitation:** because both realism scoring and precision use the same Inception
features and the same k=3 manifold, `realism(g) ≥ 1` is mathematically equivalent to
"g is inside the k-NN precision manifold". Sorting by realism and taking the top-N therefore
forces precision → 1 once N drops below the number of in-manifold images (~58% of the total).
The exact mechanism by which the paper avoids this (different k, different reference set, or a
different evaluation protocol) is still under investigation.

### `4_baselines.sh <N>`
Takes a budget `N` and evaluates three baselines, reporting FID + Precision/Recall for each:

| Baseline | Selection strategy | Feature file |
|---|---|---|
| **Random** | uniform random sample of N images (`--num N`) | `fid_features.pt` |
| **Generative Uncertainty** | N images with lowest epistemic uncertainty (`entropy_clip.npy`) | `fid_features_filtered_entropy_clip.pt` |
| **Realism** | N images with highest realism score (`realism.npy`) | `fid_features_filtered_realism.pt` |

For the GU and Realism baselines, `idx_sort.py` first writes a sorted index file
(`idx_sorted_N_<name>.npy`), then `fid.py calc --idx_path` recomputes Inception features
on just those N images.

The realism baseline **requires `realism.npy` to already exist** (produced by `3_realism_eval.sh`).
This is why the all-features and realism-eval steps run once before the per-N baseline loop.

### `gpu_job_launcher.sh`
LSF job script (DTU HPC). Sources `vars.sh`, then runs the pipeline in order:
1. `3_all_features.sh`
2. `3_realism_eval.sh`
3. `4_baselines.sh` for each N in {6000 … 12000}

---

## Why the Circular Dependency Was Broken

The previous setup shared a single `fid_features.pt` for both the random baseline and the
realism-score computation. Because the random baseline writes `fid_features.pt` with only `N`
randomly chosen images, realism scores would have been computed on a different subset for each
budget level—not on the full image set.

The fix is to decouple the two uses:
- `fid_features_all.pt` – full set, computed once, used only for the full-set FID sanity check
- `fid_features.pt` – N random images, recomputed per-N, used only for the random baseline

## Known Issue: Realism Precision Saturates to 1.0

With the current setup (Inception features, k=3 for both realism and P&R), the realism score
and the k-NN precision manifold membership are mathematically identical predicates:

$$\text{realism}(g) \geq 1 \iff \exists r: d(g,r) \leq D_r^{(k)} \iff g \in \text{manifold}$$

Filtering to the top-N by realism yields exactly the N_in_manifold highest-scoring images,
so precision = min(1, N_in_manifold / N). This is confirmed by the log: N_in_manifold ≈ 7039,
giving precision(N=7000) = 1.0.

The paper apparently avoids this saturation despite also using Inception features. The exact
structural difference (different k values for realism vs P&R, different reference set, or a
different evaluation protocol) has not yet been identified.
