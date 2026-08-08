fixed_class=10000
mc_size=5
hps_batch_size=1
m=0

# for m in $(seq 0 $((mc_size))); do
    # echo "HPSv3 scoring MC sample ${m}/$((mc_size))"
python hps_score.py \
    --img-dir "${EXP_PATH}/${m}/imgs" \
    --fixed-class $fixed_class \
    --classes-npy "${EXP_PATH}/classes.npy" \
    --hps-batch-size $hps_batch_size
# done
