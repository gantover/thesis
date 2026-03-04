# Each seed will create a new unique model allowing us to do an ensemble method to
# evaluate the generative uncertainty
for SEED in 0 1 2 3 4 5 
do
  echo "training for SEED : ${SEED}"
  python train_toy.py \
      --dataset gaussian25 \
      --size 100_000 \
      --epochs 500 \
      --exp_str "ensemble_model_seed_${SEED}_low" \
      --timesteps 1000 \
      --batch-size 10000 \
      --seed $SEED \
      --eval-intv 10000 \
      --num_sample_images 10 \
      --mid-feature 32 \
      --num-temporal-layers 2
done

# last two parameters are used to bypass the evaluation step