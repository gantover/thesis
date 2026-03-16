H=128

# Compute FID reference stats + Inception features of the real validation images
# (used for FID and final P&R evaluation in 4_baselines.sh).
python fid.py ref --data=$REF_PATH/fid-refs/imagenet-128x128.zip --dest=$REF_PATH/fid-refs/imagenet-128x128.npz --fid_features=$REF_PATH/precision-recall-refs/image_net_val_${H}_fid_features_.pt

# Extract Inception features for a 50K ImageNet *training* subset.
# These are used exclusively as the reference for scoring realism and rarity
# (3_realism_eval.sh and 3_rarity_eval.sh), decoupled from the validation
# reference used in the final P&R evaluation to prevent data leakage.
if [ ! -f $REF_PATH/fid-refs/imagenet-train-128x128.zip ]; then
    python dataset_tool.py \
        --source=${TRAIN_DATA_PATH} \
        --dest=$REF_PATH/fid-refs/imagenet-train-128x128.zip \
        --resolution=128x128 \
        --transform=center-crop \
        --max-images 50000
fi
python fid.py ref \
    --data=$REF_PATH/fid-refs/imagenet-train-128x128.zip \
    --dest=$REF_PATH/fid-refs/imagenet-train-128x128.npz \
    --fid_features=$REF_PATH/precision-recall-refs/image_net_train_${H}_fid_features_.pt