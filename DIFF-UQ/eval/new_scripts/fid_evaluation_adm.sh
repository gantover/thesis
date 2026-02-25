

# python fid.py ref --data=./fid-refs/imagenet-128x128.zip --dest=./fid-refs/imagenet-128x128.npz --fid_features=./precision-recall-refs/image_net_128_fid_features_.pt
exp_path="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234_working"
root_path="/dtu/blackhole/13/213811/s243425"
m=0
H=128
N=1000 # subset of images randomly selected
unc_name="entropy_clip"
reverse="false"

echo "random baseline for N=${N}"
python fid.py calc --images="${exp_path}/${m}/imgs" --ref=$root_path/fid-refs/imagenet-${H}x${H}.npz --num $N --fid_features="${exp_path}/${m}/fid_features.pt"
python precision_recall_torch.py --ref=$root_path/precision-recall-refs/image_net_${H}_fid_features_.pt --eval "${exp_path}/${m}/fid_features.pt"

echo "filtered G.U. baseline for N=${N}"
python idx_sort.py --path ${exp_path} --name ${unc_name} --N $N --reverse ${reverse}
echo "idx sort done, starting fid calculation"
python fid.py calc --images="${exp_path}/${m}/imgs" --ref=$root_path/fid-refs/imagenet-${H}x${H}.npz --fid_features="${exp_path}/${m}/fid_features_filtered_${unc_name}.pt" --idx_path="${exp_path}/idx_sorted_${N}_${unc_name}.npy"
echo "fid calculation done, starting precesion, recall calculation"
python precision_recall_torch.py --ref $root_path/precision-recall-refs/image_net_${H}_fid_features_.pt --eval "${exp_path}/${m}/fid_features_filtered_${unc_name}.pt"