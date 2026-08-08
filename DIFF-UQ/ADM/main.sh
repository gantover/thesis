DEVICES="0"
data="imagenet128_guided"
steps="50"
mc_size="5"
# sample_batch_size="256"
# sample_batch_size="16"
sample_batch_size="128"
total_n_sample="12032" # was 12032 but I don't have all night
# total_n_sample="1024" # was 12032 but I don't have all night
train_la_data_size="100"
train_la_batch_size="32"
# train_la_batch_size="16"
DIS="uniform"
fixed_class="10000"
seed="1234"
guidance_mode="classifier_free" # set to "classifier" to enable classifier guidance
classifier_grad_batch_size="16" # microbatch for classifier guidance gradient, lower reduces VRAM
# exp_path="./images"
exp_path="/dtu/blackhole/13/213811/s243425/images"

echo "Generating samples"
CUDA_VISIBLE_DEVICES=$DEVICES python main.py \
--config $data".yml" --timesteps=$steps --skip_type=$DIS --train_la_batch_size $train_la_batch_size \
--mc_size=$mc_size --sample_batch_size=$sample_batch_size --fixed_class=$fixed_class --train_la_data_size=$train_la_data_size \
--total_n_sample=$total_n_sample --fixed_class=$fixed_class --seed=$seed --exp_path=$exp_path --guidance_mode=$guidance_mode \
--classifier_grad_batch_size=$classifier_grad_batch_size \
--save_unet_features \
--unet_target_layers \
    output_blocks.0   output_blocks.0.0 output_blocks.0.1 \
    output_blocks.1   output_blocks.1.0 output_blocks.1.1 \
    output_blocks.2   output_blocks.2.0 output_blocks.2.1 output_blocks.2.2 \
    output_blocks.3   output_blocks.3.0 output_blocks.3.1 \
    output_blocks.4   output_blocks.4.0 output_blocks.4.1 \
    output_blocks.5   output_blocks.5.0 output_blocks.5.1 output_blocks.5.2 \
--unet_target_timesteps 45 40 35 30 25 20 15 10 5
