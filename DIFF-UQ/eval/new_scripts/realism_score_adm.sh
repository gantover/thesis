exp_path="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234_working"
root_path="/dtu/blackhole/13/213811/s243425"
m=0
H=128

python precision_recall_torch.py --ref $root_path/precision-recall-refs/image_net_${H}_fid_features_.pt --eval "${exp_path}/${m}/fid_features.pt" --realism "${exp_path}/${m}/realism.npy"