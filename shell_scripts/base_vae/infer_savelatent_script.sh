export EXP_NAME=$1
export DATA_CONFIG=$2
export MODEL_CONFIG=$3
export DATA_ROOT=$4
export SAVE_SUFFIX=$5

TRAIN_SAVE_DIR=${DATA_ROOT}/train_latent${SAVE_SUFFIX}
VALID_SAVE_DIR=${DATA_ROOT}/valid_latent${SAVE_SUFFIX}
mkdir ${TRAIN_SAVE_DIR}
mkdir ${VALID_SAVE_DIR}

NNODES=$ARNOLD_WORKER_NUM
GPUS_PER_NODE=$ARNOLD_WORKER_GPU
NODE_RANK=$ARNOLD_ID
MASTER_ADDR=$ARNOLD_WORKER_0_HOST
RAND_PORT=$(( ((RANDOM<<15)|RANDOM) % 20000 + 20000 ))
PORT=$ARNOLD_WORKER_0_PORT

# Start inference
# accelerate launch \
accelerate launch --multi_gpu \
  --machine_rank=$NODE_RANK \
  --num_machines=$NNODES \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$PORT \
  --num_processes=$[$GPUS_PER_NODE*$NNODES] \
  py_scripts/vae_inference_latentsave.py \
  --train_save_dir ${TRAIN_SAVE_DIR} \
  --valid_save_dir ${VALID_SAVE_DIR} \
  --resume="experiments/${EXP_NAME}" \
  --data_config=$DATA_CONFIG  \
  --model_config=$MODEL_CONFIG \
  --exp_root=experiments \
  --batch_size=4 \
  --mixed_precision="fp16" \
  --dataloader_num_workers=8 \
  --seed="0" 
