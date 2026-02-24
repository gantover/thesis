DEVICES="0"
data="imagenet128_guided"
steps="50"
mc_size="5"
sample_batch_size="128"
total_n_sample="1024" # was 12032 but I don't have all night
train_la_data_size="100"
train_la_batch_size="16"
DIS="uniform"
fixed_class="10000"
seed="1234"
# exp_path="./images"
exp_path="/dtu/blackhole/13/213811/s243425/images"

echo "Generating samples"
CUDA_VISIBLE_DEVICES=$DEVICES python main.py \
--config $data".yml" --timesteps=$steps --skip_type=$DIS --train_la_batch_size $train_la_batch_size \
--mc_size=$mc_size --sample_batch_size=$sample_batch_size --fixed_class=$fixed_class --train_la_data_size=$train_la_data_size \
--total_n_sample=$total_n_sample --fixed_class=$fixed_class --seed=$seed --exp_path=$exp_path
