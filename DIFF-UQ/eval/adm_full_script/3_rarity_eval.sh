H=128
m=0

# Compute a per-image rarity score for every generated image.
# Uses fid_features_all.pt (all images, from 3_all_features.sh) so that the
# resulting rarity.npy covers the full image set and its indices match those
# used later by idx_sort.py in 4_baselines.sh.
#
# Higher rarity score = the image covers a rarer (sparser) region of the real
# data manifold; inf = outside all reference k-NN balls (anomaly).
# In 4_baselines.sh we select the N images with the LOWEST rarity score
# (reverse=false), keeping the most typical/common images per the paper.
echo "rarity score calculation"
python precision_recall_torch.py \
    --ref $REF_PATH/precision-recall-refs/image_net_train_${H}_fid_features_.pt \
    --eval "${EXP_PATH}/${m}/fid_features_all.pt" \
    --rarity "${EXP_PATH}/${m}/rarity.npy"
