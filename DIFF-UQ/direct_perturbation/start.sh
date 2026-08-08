# EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_direct_perturbation_1024"
# EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234_direct_perturbation_full"
# EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234_full"
# EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234_direct_perturbation_full_fly"
# EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234_direct_perturbation_full_fly_2"
# EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_classifier_free_fixed_class10000_train%100_step50_S5_epi_unc_1234_1024_unet"
# EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_classifier_free_fixed_class10000_train%100_step50_S5_epi_unc_1234_full_unet"
EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_classifier_free_fixed_class10000_train%100_step50_S5_epi_unc_1234"
# python ./direct_perturbation/pixel_perturbation.py --exp_dir "$EXP_DIR" --m 5 --sigma 0.02
# python ./direct_perturbation/noise_transform.py --exp_dir "$EXP_DIR" --m 20 --sigma 0.02
# python semantic_likelihood.py --path "$EXP_DIR" --encoder "clip"
# python compute_uncertainty.py --path "$EXP_DIR" --mode onthefly --M 20 --sigma 0.02 --use_transforms --encoder clip 
# python compute_uncertainty.py --path "$EXP_DIR" --mode onthefly --M 128 --sigma 0.02 --encoder clip --entropy-calculation distance
# python compute_uncertainty.py --path "$EXP_DIR" --mode onthefly --M 5 --sigma 0.02 --encoder dinov2_vitl14_reg --entropy-calculation diagonal --unanchored-variance
# python compute_uncertainty.py --path "$EXP_DIR" --mode onthefly --M 10 --sigma 0.02 --encoder dinov2_vitl14_reg --entropy-calculation trace --unanchored-variance
FEATURES_DIR="$EXP_DIR/0/features"
mapfile -t KEYS < <(ls "$FEATURES_DIR" | sort)
TOTAL=${#KEYS[@]}

echo "Found $TOTAL feature keys in $FEATURES_DIR"

for i in "${!KEYS[@]}"; do
    KEY="${KEYS[$i]}"
    echo "[$(( i + 1 ))/$TOTAL] Computing uncertainty for $KEY"
    python compute_uncertainty.py --path "$EXP_DIR" --mode precomputed --M 6 \
        --encoder unet_internal --unet-feature-key "$KEY" \
        --entropy-calculation diagonal --unanchored-variance
done