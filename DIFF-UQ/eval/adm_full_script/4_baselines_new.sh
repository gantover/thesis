#!/bin/bash
# Step 4: Compute baselines (Random, G.U., Realism) for a given N.
#
# PREREQUISITE: Step 3 must have been run first, producing:
#   - ${EXP_PATH}/${m}/fid_features_all.pt  (Inception features for ALL images)
#   - ${EXP_PATH}/${m}/realism.npy           (realism score per image)
#
# This script does NOT need a GPU — it works entirely from pre-computed
# Inception features, avoiding the circular dependency where:
#   - realism baseline needs fid_features for ALL images
#   - but fid.py calc would overwrite them with a subset
#
# Usage: bash 4_baselines.sh <N>
#   where N is the subset size to evaluate

set -e

N=$1
if [ -z "$N" ]; then
    echo "Usage: bash 4_baselines.sh <N>"
    exit 1
fi

unc_name="entropy_clip"
FEATURES_ALL="${EXP_PATH}/${m}/fid_features_all.pt"
REF_STATS="${ROOT_PATH}/fid-refs/imagenet-${H}x${H}.npz"
REF_FEATURES="${ROOT_PATH}/precision-recall-refs/image_net_val_${H}_fid_features_.pt"

# Verify prerequisites
if [ ! -f "$FEATURES_ALL" ]; then
    echo "ERROR: $FEATURES_ALL not found. Run step 3 first."
    exit 1
fi

echo "========================================"
echo "  Baselines for N=${N}"
echo "========================================"

# ── 1. Random baseline ─────────────────────────────────────────────────────
echo ""
echo "=== Random baseline (N=${N}) ==="
# Randomly select N images from pre-computed features and compute FID
python fid.py calc-from-features \
    --features="$FEATURES_ALL" \
    --ref="$REF_STATS" \
    --num=$N \
    --save_features="${EXP_PATH}/${m}/fid_features_random_${N}.pt"

# Precision/Recall for the random subset
python precision_recall_torch.py \
    --ref="$REF_FEATURES" \
    --eval="${EXP_PATH}/${m}/fid_features_random_${N}.pt"


# ── 2. Generative Uncertainty (G.U.) baseline ──────────────────────────────
echo ""
echo "=== G.U. baseline: ${unc_name} (N=${N}) ==="
# Sort by uncertainty score (ascending = keep most certain)
python idx_sort.py --path ${EXP_PATH} --name ${unc_name} --N $N --reverse false

# FID from pre-computed features using the sorted indices
python fid.py calc-from-features \
    --features="$FEATURES_ALL" \
    --ref="$REF_STATS" \
    --idx_path="${EXP_PATH}/idx_sorted_${N}_${unc_name}.npy" \
    --save_features="${EXP_PATH}/${m}/fid_features_filtered_${unc_name}_${N}.pt"

# Precision/Recall for the G.U. subset
python precision_recall_torch.py \
    --ref="$REF_FEATURES" \
    --eval="${EXP_PATH}/${m}/fid_features_filtered_${unc_name}_${N}.pt"


# ── 3. Realism baseline ────────────────────────────────────────────────────
echo ""
echo "=== Realism baseline (N=${N}) ==="
# Sort by realism score (descending = keep most realistic)
python idx_sort.py --path ${EXP_PATH}/${m} --name realism --N $N --reverse true

# FID from pre-computed features using the sorted indices
python fid.py calc-from-features \
    --features="$FEATURES_ALL" \
    --ref="$REF_STATS" \
    --idx_path="${EXP_PATH}/${m}/idx_sorted_${N}_realism.npy" \
    --save_features="${EXP_PATH}/${m}/fid_features_filtered_realism_${N}.pt"

# Precision/Recall for the realism subset
python precision_recall_torch.py \
    --ref="$REF_FEATURES" \
    --eval="${EXP_PATH}/${m}/fid_features_filtered_realism_${N}.pt"


echo ""
echo "=== All baselines for N=${N} done ==="
