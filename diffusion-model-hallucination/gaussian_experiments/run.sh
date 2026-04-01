# Each seed will create a new unique model allowing us to do an ensemble method to
# evaluate the generative uncertainty
for SEED in 0 1 2 3 4 5 
do
  echo "training for SEED : ${SEED}"
  python train_toy.py \
      --dataset gaussian25 \
      --size 100_000 \
      --epochs 10000 \
      --generations 1 \
      --exp_str "s1_fds_ensemble_model_seed_$SEED" \
      --timesteps 1000 \
      --batch-size 10000 \
      --seed $SEED \
      --dataset-seed 21 \
      --eval-intv 10000 \
      --num_sample_images 10
done

# last two parameters are used to bypass the evaluation step