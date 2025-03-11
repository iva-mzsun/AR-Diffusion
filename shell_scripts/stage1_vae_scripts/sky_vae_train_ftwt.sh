export DATA_CONFIG="configs/sky/data/skyvid.yaml"
export MODEL_CONFIG="configs/sky/vae/vae_4ch_32len_large_wt.yaml"
export EXP_SUFFIX=""
export WANDB_API_KEY=""

# Start training
NNODES=$ARNOLD_WORKER_NUM
GPUS_PER_NODE=$ARNOLD_WORKER_GPU
NODE_RANK=$ARNOLD_ID
MASTER_ADDR=$ARNOLD_WORKER_0_HOST
RAND_PORT=$(( ((RANDOM<<15)|RANDOM) % 20000 + 20000 ))
PORT=$ARNOLD_WORKER_0_PORT

accelerate launch --multi_gpu \
  --machine_rank=$NODE_RANK \
    --num_machines=$NNODES \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$PORT \
    --num_processes=$[$GPUS_PER_NODE*$NNODES] \
  py_scripts/vae_train_kl.py \
  --suffix=$EXP_SUFFIX \
  --data_config=$DATA_CONFIG  \
  --model_config=$MODEL_CONFIG \
    --exp_root=experiments \
  --mixed_precision="fp16" \
  --batch_size=1 \
  --num_train_epochs=1000 \
  --validation_steps=1000 \
  --checkpointing_steps=2500 \
  --dataloader_num_workers=4 \
  --gradient_accumulation_steps=4 \
  --kl_loss_weight=0.0 \
  --learning_rate=1e-6 \
  --lr_warmup_steps=0 \
  --lr_scheduler="constant" \
  --seed="0" \
  --skip_initial_model \
  --discriminator_loss_start_step=-1 \
  --display_col=1 \
  --trainable_module_key=temporal \
  --report_to="wandb" \
  --resume_skip_optimizer \
  --pretrained_ckpt=experiments/skytimelapse_vae_wot/ckpt_dir/checkpoint-50000
