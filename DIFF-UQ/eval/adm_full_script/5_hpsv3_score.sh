# Generation parameters — must match ADM/main.sh exactly so the RNG replay
# reproduces the correct class assignments for each image.
seed=1234
total_n_sample=12032
sample_batch_size=256
image_size=128
channels=3
fixed_class=10000
mc_size=5
hps_batch_size=1
m=0

# for m in $(seq 0 $((mc_size))); do
    # echo "HPSv3 scoring MC sample ${m}/$((mc_size))"
python hps_score.py \
    --img-dir "${EXP_PATH}/${m}/imgs" \
    --seed $seed \
    --total-n-sample $total_n_sample \
    --sample-batch-size $sample_batch_size \
    --image-size $image_size \
    --channels $channels \
    --fixed-class $fixed_class \
    --hps-batch-size $hps_batch_size
# done
