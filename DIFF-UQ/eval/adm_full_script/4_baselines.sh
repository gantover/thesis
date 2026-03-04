m=0
H=128
N=$1 # subset of images randomly selected
unc_name="entropy_clip"
reverse="false"

echo "random baseline for N=${N}"
python fid.py calc --images="${EXP_PATH}/${m}/imgs" --ref=$ROOT_PATH/fid-refs/imagenet-${H}x${H}.npz --num $N --fid_features="${EXP_PATH}/${m}/fid_features.pt"
python precision_recall_torch.py --ref=$ROOT_PATH/precision-recall-refs/image_net_val_${H}_fid_features_.pt --eval "${EXP_PATH}/${m}/fid_features.pt"

echo "filtered G.U. baseline for N=${N}"
python idx_sort.py --path ${EXP_PATH} --name ${unc_name} --N $N --reverse ${reverse}
echo "idx sort done, starting fid calculation"
python fid.py calc --images="${EXP_PATH}/${m}/imgs" --ref=$ROOT_PATH/fid-refs/imagenet-${H}x${H}.npz --fid_features="${EXP_PATH}/${m}/fid_features_filtered_${unc_name}.pt" --idx_path="${EXP_PATH}/idx_sorted_${N}_${unc_name}.npy"
echo "fid calculation done, starting precesion, recall calculation"
python precision_recall_torch.py --ref $ROOT_PATH/precision-recall-refs/image_net_val_${H}_fid_features_.pt --eval "${EXP_PATH}/${m}/fid_features_filtered_${unc_name}.pt"

# exp_name="realism"

# echo "filtered realism score baseline for N=${N}"
# python idx_sort.py --path ${EXP_PATH}/${m} --name ${exp_name} --N $N --reverse true
# echo "idx sort done, starting fid calculation"
# python fid.py calc --images="${EXP_PATH}/${m}/imgs" --ref=$ROOT_PATH/fid-refs/imagenet-${H}x${H}.npz --fid_features="${EXP_PATH}/${m}/fid_features_filtered_${exp_name}.pt" --idx_path="${EXP_PATH}/${m}/idx_sorted_${N}_${exp_name}.npy"
# echo "fid calculation done, starting precesion, recall calculation"
# python precision_recall_torch.py --ref $ROOT_PATH/precision-recall-refs/image_net_val_${H}_fid_features_.pt --eval "${EXP_PATH}/${m}/fid_features_filtered_${exp_name}.pt"