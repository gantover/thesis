# Each seed will create a new unique model allowing us to do an ensemble method to
# evaluate the generative uncertainty
for SEED in 0 1 2 3 4 5 
do
  echo "training for SEED : ${SEED}"
  python train_toy.py \
      --dataset gaussian25 \
      --size 25000 \
      --epochs 2500 \
      --generations 1 \
      --exp_str "quarter_fixed_ds_ensemble_model_seed_$SEED" \
      --timesteps 1000 \
      --batch-size 2500 \
      --seed $SEED \
      --dataset-seed 21 \
      --eval-intv 2500 \
      --num_sample_images 10
done