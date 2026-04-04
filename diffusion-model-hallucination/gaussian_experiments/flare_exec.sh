/dtu/blackhole/13/213811/s243425/thesis/.venv/bin/python run.py generate-flare-scores-single \
  --checkpoint-path chkpts/gaussian25_100000_g_1_e_10000_t1000_m128_nl3_blinear_seed0_fixed_ds_ensemble_model_seed_0/ddpm_gaussian25_gen_0.pt \
  --real-data-path chkpts/gaussian25_100000_g_1_e_10000_t1000_m128_nl3_blinear_seed0_fixed_ds_ensemble_model_seed_0/real_dataset.npy \
  --output-dir /dtu/blackhole/13/213811/s243425/gaussian_experiment/samples/flare \
  --n-score-samples 8096 \
  --subset random \
  --m 512 \
  --prior-precision 400.0 \
  --max-posterior-std 1.0 \
  --fit-batches 64 \
  --fit-batch-size 512 \
  --score-batch-size 128 \
  --tail-steps 100
  