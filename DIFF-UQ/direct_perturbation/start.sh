# EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_direct_perturbation_1024"
# EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234_direct_perturbation_full"
# EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234_full"
# EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234_direct_perturbation_full_fly"
EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234_direct_perturbation_full_fly_2"
# python ./direct_perturbation/pixel_perturbation.py --exp_dir "$EXP_DIR" --m 5 --sigma 0.02
# python ./direct_perturbation/noise_transform.py --exp_dir "$EXP_DIR" --m 20 --sigma 0.02
# python semantic_likelihood.py --path "$EXP_DIR" --encoder "clip"
# python compute_uncertainty.py --path "$EXP_DIR" --mode onthefly --M 20 --sigma 0.02 --use_transforms --encoder clip 
# python compute_uncertainty.py --path "$EXP_DIR" --mode onthefly --M 128 --sigma 0.02 --encoder clip --entropy-calculation distance
python compute_uncertainty.py --path "$EXP_DIR" --mode onthefly --M 8 --sigma 0.02 --encoder dinov2_vits14_reg --entropy-calculation diagonal --unanchored-variance