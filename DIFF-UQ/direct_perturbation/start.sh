# EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_direct_perturbation_1024"
# EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234_direct_perturbation_full"
EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234_full"
# python ./direct_perturbation/pixel_perturbation.py --exp_dir "$EXP_DIR" --m 5 --sigma 0.02
python semantic_likelihood.py --path "$EXP_DIR" --encoder "clip"