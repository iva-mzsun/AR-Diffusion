export DATA_CONFIG="configs/openvid/data/openvidimg.yaml"
export MODEL_CONFIG="configs/openvid/vae/vae_4ch_32len_large.yaml"
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
  py_scripts/vae_train_kl_disc.py \
  --suffix=$EXP_SUFFIX \
  --data_config=$DATA_CONFIG  \
  --model_config=$MODEL_CONFIG \
  --disc_factor=1.0 \
  --exp_root=experiments \
  --mixed_precision="fp16" \
  --batch_size=16 \
  --num_train_epochs=35000 \
  --validation_steps=1000 \
  --checkpointing_steps=2500 \
  --dataloader_num_workers=2 \
  --gradient_accumulation_steps=2 \
  --kl_loss_weight=0.0 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=0 \
  --lr_scheduler="constant" \
  --seed="0" \
  --max_train_steps=500000 \
  --discriminator_loss_start_step=50000 \
  --display_col=4 \
  --gradient_clip_norm=1.0 \
  --report_to="wandb"
