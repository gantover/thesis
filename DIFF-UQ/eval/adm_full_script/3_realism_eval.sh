H=128
m=0
echo "realism score calculation"
python precision_recall_torch.py --ref $ROOT_PATH/precision-recall-refs/image_net_val_${H}_fid_features_.pt --eval "${EXP_PATH}/${m}/fid_features.pt" --realism "${EXP_PATH}/${m}/realism.npy"