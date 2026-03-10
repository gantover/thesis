m=0
H=128

# Extract Inception-v3 features for ALL generated images without any count limit.
# Saved as fid_features_all.pt; this file is the prerequisite for realism score
# computation (3_realism_eval.sh) and must exist before running 4_baselines.sh.
echo "Extracting Inception features for all images (no size cap)"
python fid.py calc \
    --images="${EXP_PATH}/${m}/imgs" \
    --ref=$ROOT_PATH/fid-refs/imagenet-${H}x${H}.npz \
    --fid_features="${EXP_PATH}/${m}/fid_features_all.pt" \
    --num 0
