m=0
H=128
N=$1 # subset of images randomly selected
unc_name="entropy_clip"
reverse="false"

# --- Random baseline ---
# Sample N images uniformly at random and compute FID + precision/recall.
echo "random baseline for N=${N}"
python fid.py calc --images="${EXP_PATH}/${m}/imgs" --ref=$ROOT_PATH/fid-refs/imagenet-${H}x${H}.npz --num $N --fid_features="${EXP_PATH}/${m}/fid_features.pt"
python precision_recall_torch.py --ref $ROOT_PATH/precision-recall-refs/image_net_val_${H}_fid_features_.pt --eval "${EXP_PATH}/${m}/fid_features.pt"

# --- Generative Uncertainty (GU) baseline ---
# Keep the N images with the lowest epistemic uncertainty score.
echo "filtered G.U. baseline for N=${N}"
python idx_sort.py --path ${EXP_PATH} --name ${unc_name} --N $N --reverse ${reverse}
echo "idx sort done, starting fid calculation"
python fid.py calc --images="${EXP_PATH}/${m}/imgs" --ref=$ROOT_PATH/fid-refs/imagenet-${H}x${H}.npz --fid_features="${EXP_PATH}/${m}/fid_features_filtered_${unc_name}.pt" --idx_path="${EXP_PATH}/idx_sorted_${N}_${unc_name}.npy"
echo "fid calculation done, starting precision/recall calculation"
python precision_recall_torch.py --ref $ROOT_PATH/precision-recall-refs/image_net_val_${H}_fid_features_.pt --eval "${EXP_PATH}/${m}/fid_features_filtered_${unc_name}.pt"

# --- Realism baseline ---
# Keep the N images with the highest realism score (pre-computed in 3_realism_eval.sh).
# Sorting in reverse (descending) selects the most realistic images first.
echo "filtered realism score baseline for N=${N}"
python idx_sort.py --path ${EXP_PATH}/${m} --name realism --N $N --reverse true
echo "idx sort done, starting fid calculation"
python fid.py calc --images="${EXP_PATH}/${m}/imgs" --ref=$ROOT_PATH/fid-refs/imagenet-${H}x${H}.npz --fid_features="${EXP_PATH}/${m}/fid_features_filtered_realism.pt" --idx_path="${EXP_PATH}/${m}/idx_sorted_${N}_realism.npy"
echo "fid calculation done, starting precision/recall calculation"
python precision_recall_torch.py --ref $ROOT_PATH/precision-recall-refs/image_net_val_${H}_fid_features_.pt --eval "${EXP_PATH}/${m}/fid_features_filtered_realism.pt"