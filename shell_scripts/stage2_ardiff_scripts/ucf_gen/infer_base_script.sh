TRAINDATA_FILE=$1
VALIDDATA_FILE=$2
MODEL_FILE_NAME=$3
export DATA_CONFIG=configs/ucf101/data/${VALIDDATA_FILE}.yaml
export MODEL_CONFIG=configs/ucf101/ardiff/${MODEL_FILE_NAME}.yaml
# export EXP_SUFFIX=$3
export EXP_SUFFIX=${MODEL_FILE_NAME}_${TRAINDATA_FILE}
export UCGS=$4
export NUM_FRAMES=$5
export ARDIFF_STEP=$6

# Find target exp
cd experiments
files=$( ls -d *${EXP_SUFFIX}* | sort )
EXP_NAME=$(echo "$files" | tail -n 1)
cd ..

# Print settings
echo EXP name: ${EXP_NAME}
echo ardiff step: ${ARDIFF_STEP}

# Obtain the resume path
if [ -z "$CKPT_STEP" ]; then
  RESUME_PATH="experiments/${EXP_NAME}"
else
  RESUME_PATH="experiments/${EXP_NAME}/ckpt_dir/checkpoint-${CKPT_STEP}"
fi
echo $RESUME_PATH

# Obtain save_dir
SAVE_DIR=save_ucgs${UCGS}_numframes${NUM_FRAMES}_ardiffstep${ARDIFF_STEP}

# Start inference
NNODES=$ARNOLD_WORKER_NUM
GPUS_PER_NODE=$ARNOLD_WORKER_GPU
NODE_RANK=$ARNOLD_ID
MASTER_ADDR=$ARNOLD_WORKER_0_HOST
RAND_PORT=$(( ((RANDOM<<15)|RANDOM) % 20000 + 20000 ))
# PORT=$ARNOLD_WORKER_0_PORT
PORT=11234
accelerate config default
export NCCL_DEBUG=WARN

# ======= Generate Samples ======
# accelerate launch --multi_gpu \
# --machine_rank=$NODE_RANK \
# --num_machines=$NNODES \
# --main_process_ip=$MASTER_ADDR \
# --main_process_port=$PORT \
# --num_processes=$[$GPUS_PER_NODE*$NNODES] \
accelerate launch \
  py_scripts/diff_inference.py \
  \
  --resume="${RESUME_PATH}" \
  --data_config=$DATA_CONFIG  \
  --model_config=$MODEL_CONFIG \
  --ucgs=${UCGS} \
  --num_frames=${NUM_FRAMES} \
  --ardiff_step=${ARDIFF_STEP} \
  --save_dir=${SAVE_DIR} \
  --exp_root=experiments \
  --batch_size=1 \
  --sample_num=256 \
  --mixed_precision="fp16" \
  --dataloader_num_workers=8 \
  --seed="0" \
  # --verbose

# ======= Calculate Metric ======
# EXP_FOLDER=experiments/${EXP_NAME}
# GT_DIR=${EXP_FOLDER}/$SAVE_DIR/test_gt
# PRED_DIR=${EXP_FOLDER}/$SAVE_DIR/test_pred
# ROOT_DIR=$PWD

# # sample without conditional guidance
# python3  tool/metrics/metric_center.py \
# --root_dir ${ROOT_DIR} \
# --path_gen ${PRED_DIR} --path_gt ${GT_DIR} --type fid-img fid-vid fvd  \
#  --write_metric_to ${EXP_FOLDER}/logs/metrics_${SAVE_DIR}.json --number_sample_frames ${NUM_FRAMES} --sample_duration ${NUM_FRAMES}

# Delete samples
# rm -rf ${EXP_FOLDER}/$SAVE_DIR
