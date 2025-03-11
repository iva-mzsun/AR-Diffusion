export DATA_CONFIG="configs/OpenVid/data/openvidimg.yaml"
export MODEL_CONFIG="configs/OpenVid/model/basel1.yaml"
export EXP_SUFFIX="-stage1"
export WANDB_API_KEY="d0975e6993188803317cf49b8cae27d6a1eeb8b7"

# Start training
NNODES=$ARNOLD_WORKER_NUM
GPUS_PER_NODE=$ARNOLD_WORKER_GPU
NODE_RANK=$ARNOLD_ID
MASTER_ADDR=$ARNOLD_WORKER_0_HOST
RAND_PORT=$(( ((RANDOM<<15)|RANDOM) % 20000 + 20000 ))
PORT=$ARNOLD_WORKER_0_PORT

export NCCL_DEBUG=INFO

# accelerate launch \
accelerate launch --multi_gpu \
--machine_rank=$NODE_RANK \
--num_machines=$NNODES \
--main_process_ip=$MASTER_ADDR \
--main_process_port=$PORT \
--num_processes=$[$GPUS_PER_NODE*$NNODES] \
py_scripts/vae_train.py \
--suffix=$EXP_SUFFIX \
--data_config=$DATA_CONFIG  \
--model_config=$MODEL_CONFIG \
--exp_root=experiments \
--mixed_precision="fp16" \
--batch_size=15 \
--num_train_epochs=1000 \
--validation_steps=1000 \
--checkpointing_steps=2500 \
--dataloader_num_workers=2 \
--gradient_accumulation_steps=8 \
--learning_rate=2e-4 \
--lr_warmup_steps=0 \
--lr_scheduler="constant" \
--seed="0" \
--display_col=5 \
--resume=experiments/2024-08-29T17-44-43_basel1_openvidimg-stage1 \
--report_to="wandb" \

# --resume_skip_optimizer \
# --resume_skip_discriminator \
# --resume=experiments/2024-07-23T14-24-15_basel1_ucfimg
# --trainable_module_key temporal \
# --resume_skip_optimizer \
# --resume_skip_discriminator \
# --pretrained_ckpt=experiments/2024-07-12T15-00-56_basel1_ucfimg/ckpt_dir/checkpoint-105000
