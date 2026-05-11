data_path="./data"
save_root="/DATA/DrugJEPA/train_logs/DrugJEPA_train"
#save_name=""
save_dir="${save_root}/ckpt"
tmp_save_dir="${save_root}/tmp_save_dir_screen"
tsb_dir="${save_root}/tsb_dir_screen"
mkdir -p ${save_dir}
mkdir -p "${save_root}/train_log"

n_gpu=4
MASTER_PORT=10062
finetune_mol_model="./ckpt/unimol_pretrained/mol_pre_no_h_220816.pt" # unimol pretrained mol model
finetune_pocket_model="./ckpt/unimol_pretrained/pocket_pre_220816.pt" # unimol pretrained pocket model


batch_size=48
batch_size_valid=48
max_lignum=16
epoch=60
dropout=0.0
warmup=0.06
update_freq=1
dist_threshold=8.0
recycling=3
lr=1e-4

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
#export TORCH_FORCE_WEIGHTS_ONLY=false
#export TORCH_DISTRIBUTED_DETAIL=DEBUG
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --use-env --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 0 --ddp-backend=c10d --rank-loss True \
       --task train_drugjepa_moe --loss triple_moe_contrast --arch drug_jepa_moe --moe-arch alternating --use-sequence True \
       --max-pocket-atoms 256 --l2-pkt2mol True --l2-sequence False --l2-loss True \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size --batch-size-valid $batch_size_valid \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq $update_freq \
       --tensorboard-logdir $tsb_dir \
       --num-experts 8 \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --best-checkpoint-metric valid_bedroc --patience 2000 --all-gather-list-size 2048000 \
       --save-dir $save_dir --tmp-save-dir $tmp_save_dir --keep-best-checkpoints 10 --keep-last-epochs 20 \
       --find-unused-parameters \
       --maximize-best-checkpoint-metric \
       --finetune-pocket-model $finetune_pocket_model \
       --finetune-mol-model $finetune_mol_model \
       --valid-set CASF \
       --max-lignum $max_lignum \
       --protein-similarity-thres 1.0 > ${save_root}/train_log/train_log.txt
