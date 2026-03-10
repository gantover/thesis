H=128

# Compute FID reference stats + Inception features of the real images (used for P&R eval and realism scoring).
python fid.py ref --data=$ROOT_PATH/fid-refs/imagenet-128x128.zip --dest=$ROOT_PATH/fid-refs/imagenet-128x128.npz --fid_features=$ROOT_PATH/precision-recall-refs/image_net_val_${H}_fid_features_.pt