

# python fid.py ref --data=./fid-refs/imagenet-128x128.zip --dest=./fid-refs/imagenet-128x128.npz --fid_features=./precision-recall-refs/image_net_128_fid_features_.pt
exp_path="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234_working"
root_path="/dtu/blackhole/13/213811/s243425"
m=0
H=128
N=600 # subset of images randomly selected

exp_name="realism"

echo "filtered realism score baseline for N=${N}"
python idx_sort.py --path ${exp_path}/${m} --name ${exp_name} --N $N --reverse true
echo "idx sort done, starting fid calculation"
python fid.py calc --images="${exp_path}/${m}/imgs" --ref=$root_path/fid-refs/imagenet-${H}x${H}.npz --fid_features="${exp_path}/${m}/fid_features_filtered_${exp_name}.pt" --idx_path="${exp_path}/${m}/idx_sorted_${N}_${exp_name}.npy"
echo "fid calculation done, starting precesion, recall calculation"
python precision_recall_torch.py --ref $root_path/precision-recall-refs/image_net_${H}_fid_features_.pt --eval "${exp_path}/${m}/fid_features_filtered_${exp_name}.pt"
