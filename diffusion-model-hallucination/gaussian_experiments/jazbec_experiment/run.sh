# Each seed will create a new unique model allowing us to do an ensemble method to
# evaluate the generative uncertainty
for SEED in 0 1 2 3 4 5 
do
  echo "training for SEED : ${SEED}"
  python train_toy.py \
      --dataset gaussian25 \
      --size 100_000 \
      --epochs 10000 \
      --generations 5 \
      --exp_str "ensemble_model_seed_$SEED" \
      --timesteps 1000 \
      --batch-size 10000 \
      --seed $SEED \
      --eval-intv 10000 \
      --num_sample_images 10
done

# last two parameters are used to bypass the evaluation step