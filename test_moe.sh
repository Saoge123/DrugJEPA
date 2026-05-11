batch_size=256

TASK=${1}
arch=${2}
moe_arch=${3}
weight_path=${4}
device_id=${5}


#mkdir -p $results_path
python ./unimol/test.py "./test_datasets" --user-dir ./unimol --valid-subset \
  test --results-path "./test_results/" --num-workers 2 --ddp-backend=c10d \
  --batch-size 256 --task test_task --loss triple_moe_contrast \
  --arch $arch --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
  --seed 1 --device-id $device_id --moe-arch $moe_arch \
  --path $weight_path --log-interval 100 --log-format simple \
  --max-pocket-atoms 2048 --test-task $TASK