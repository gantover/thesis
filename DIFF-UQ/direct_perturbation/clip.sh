EXP_DIR="/dtu/blackhole/13/213811/s243425/images/IMAGENET128/ddim_classifier_free_fixed_class10000_train%100_step50_S5_epi_unc_1234"

python compute_uncertainty.py --path "$EXP_DIR" --mode precomputed --M 6 \
    --encoder clip \
    --entropy-calculation diagonal --unanchored-variance