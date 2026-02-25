DATA_PATH="/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/train/"
OUTPUT_ROOT="/dtu/blackhole/13/213811/s243425"

python dataset_tool.py --source=$DATA_PATH --dest=$OUTPUT_ROOT/fid-refs/imagenet-128x128.zip --resolution=128x128 --transform=center-crop --max-images 50000