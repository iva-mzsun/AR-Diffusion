DATA_FILE_NAME=$1
MODEL_FILE_NAME=$2
PRETRAINED_CKPT=$3
export DATA_CONFIG=configs/faceforensic/data/${DATA_FILE_NAME}.yaml
export MODEL_CONFIG=configs/faceforensic/ardiff/${MODEL_FILE_NAME}.yaml
export EXP_SUFFIX="_ft"
export WANDB_API_KEY=""

# Start training
NNODES=$ARNOLD_WORKER_NUM
GPUS_PER_NODE=$ARNOLD_WORKER_GPU
NODE_RANK=$ARNOLD_ID
MASTER_ADDR=$ARNOLD_WORKER_0_HOST
RAND_PORT=$(( ((RANDOM<<15)|RANDOM) % 20000 + 20000 ))
PORT=$ARNOLD_WORKER_0_PORT
export NCCL_DEBUG=WARN

# ==== Train Model ====
# accelerate launch \
# export TORCH_DISTRIBUTED_DEBUG=DETAIL 
# accelerate launch --multi_gpu \
# --machine_rank=$NODE_RANK \
# --num_machines=$NNODES \
# --main_process_ip=$MASTER_ADDR \
# --main_process_port=$PORT \
# --num_processes=$[$GPUS_PER_NODE*$NNODES] \
#    py_scripts/diff_train.py \
  --skip_vae --use_ema --offload_ema \
  --suffix=$EXP_SUFFIX \
  --data_config=$DATA_CONFIG  \
  --model_config=$MODEL_CONFIG \
  --exp_root=experiments \
  --mixed_precision="fp16" \
  --train_batch_size=16 \
  --valid_batch_size=16 \
  --num_train_epochs=100000 \
  --validation_steps=1000 \
  --checkpointing_steps=5000 \
  --checkpoints_total_limit=2 \
  --dataloader_num_workers=8 \
  --gradient_accumulation_steps=8 \
  --learning_rate=1e-5 \
  --lr_warmup_steps=0 \
  --max_train_steps=50000 \
  --lr_scheduler="constant" \
  --gradient_clip_norm=1.0 \
  --seed="0" \
  --report_to="wandb" \
  --resume_skip_optimizer \
  --pretrained_ckpt=${PRETRAINED_CKPT}

# === Infer & Calculate Metric ===
VALIDDATA_FILE=facevidflatten
TRAINDATA_FILE=$DATA_FILE_NAME
bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 16 0
bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 16 0
# bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 3.0 16 0
# bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 4.0 16 0
# bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 5.0 16 0
# bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 6.0 16 0
# bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 32 0
# bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 64 0
# bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 128 0
bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 16 5
bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 16 10
bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 16 15
bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 16 20
bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 16 25
bash shell_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 16 50

# === Occupy the current gpus ===
cd shell_scripts/gpu_use
bash burn_gpu.sh
