H=128
m=0

# Compute a per-image realism score for every generated image.
# Uses fid_features_all.pt (all images, from 3_all_features.sh) so that the
# resulting realism.npy covers the full image set and its indices match those
# used later by idx_sort.py in 4_baselines.sh.
echo "realism score calculation"
python precision_recall_torch.py \
    --ref $ROOT_PATH/precision-recall-refs/image_net_val_${H}_fid_features_.pt \
    --eval "${EXP_PATH}/${m}/fid_features_all.pt" \
    --realism "${EXP_PATH}/${m}/realism.npy"