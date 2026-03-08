# Each seed will create a new unique model allowing us to do an ensemble method to
# evaluate the generative uncertainty
for SEED in 0 1 2 3 4 5 
do
  echo "training for SEED : ${SEED}"
  python train_toy.py \
      --dataset gaussian25 \
      --size 10000 \
      --epochs 5000 \
      --generations 1 \
      --exp_str "ensemble_model_seed_$SEED" \
      --timesteps 1000 \
      --batch-size 5000 \
      --seed $SEED \
      --eval-intv 5000 \
      --num_sample_images 10
done

# last two parameters are used to bypass the evaluation step
# size was 100_000
# epochs was 10000