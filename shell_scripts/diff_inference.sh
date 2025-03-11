export DATA_CONFIG="configs/skytimelapse/data/skyvidflatten.yaml"
export MODEL_CONFIG=$1
export EXP_NAME=$2
export UCGS=$3

# Obtain the resume path
if [ -z "$CKPT_STEP" ]; then
  RESUME_PATH="experiments/${EXP_NAME}"
else
  RESUME_PATH="experiments/${EXP_NAME}/ckpt_dir/checkpoint-${CKPT_STEP}"
fi
echo $RESUME_PATH
# Obtain save_dir
SAVE_DIR=save${3}_vid

accelerate config default

# Start inference
NNODES=$ARNOLD_WORKER_NUM
GPUS_PER_NODE=$ARNOLD_WORKER_GPU
NODE_RANK=$ARNOLD_ID
MASTER_ADDR=$ARNOLD_WORKER_0_HOST
RAND_PORT=$(( ((RANDOM<<15)|RANDOM) % 20000 + 20000 ))
PORT=$ARNOLD_WORKER_0_PORT

# ======= Generate Samples ======
# accelerate launch --multi_gpu \
#   --machine_rank=$NODE_RANK \
#   --num_machines=$NNODES \
#   --main_process_ip=$MASTER_ADDR \
#   --main_process_port=$PORT \
#   --num_processes=$[$GPUS_PER_NODE*$NNODES] \
accelerate launch \
  py_scripts/diff_inference.py \
  \
  --resume="${RESUME_PATH}" \
  --data_config=$DATA_CONFIG  \
  --model_config=$MODEL_CONFIG \
  --ucgs=$UCGS \
  --save_dir=$SAVE_DIR \
  --exp_root=experiments \
  --batch_size=4 \
  --sample_num=2048 \
  --mixed_precision="fp16" \
  --dataloader_num_workers=8 \
  --seed="0" 

# ======= Calculate Metric ======
EXP_FOLDER=experiments/${EXP_NAME}
GT_DIR=${EXP_FOLDER}/$SAVE_DIR/test_gt
PRED_DIR=${EXP_FOLDER}/$SAVE_DIR/test_pred

# sample without conditional guidance
python3  tool/metrics/metric_center.py \
--root_dir /mnt/bn/lq-mzsun-arnold/codes/video_token/3d_tokenizer \
--path_gen ${PRED_DIR} --path_gt ${GT_DIR} --type fid-img fid-vid fvd  \
 --write_metric_to ${EXP_FOLDER}/logs/metrics_ucgs${UCGS}.json --number_sample_frames 16 --sample_duration 16

# Delete samples
rm -rf ${EXP_FOLDER}/$SAVE_DIR