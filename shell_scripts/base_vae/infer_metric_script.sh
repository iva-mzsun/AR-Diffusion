export EXP_NAME=$1
export DATA_CONFIG=$2
export MODEL_CONFIG=$3

NNODES=$ARNOLD_WORKER_NUM
GPUS_PER_NODE=$ARNOLD_WORKER_GPU
NODE_RANK=$ARNOLD_ID
MASTER_ADDR=$ARNOLD_WORKER_0_HOST
RAND_PORT=$(( ((RANDOM<<15)|RANDOM) % 20000 + 20000 ))
PORT=$ARNOLD_WORKER_0_PORT

# Start inference
accelerate launch --multi_gpu \
  --machine_rank=$NODE_RANK \
  --num_machines=$NNODES \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$PORT \
  --num_processes=$[$GPUS_PER_NODE*$NNODES] \
  py_scripts/vae_inference.py \
  --resume="experiments/${EXP_NAME}" \
  --data_config=$DATA_CONFIG  \
  --model_config=$MODEL_CONFIG \
  --exp_root=experiments \
  --batch_size=4 \
  --mixed_precision="fp16" \
  --dataloader_num_workers=8 \
  --seed="0" 

# ========= Start calculate metrics =========
EXP_FOLDER=experiments/${EXP_NAME}
mkdir ${EXP_FOLDER}/logs
PRED_DIR=${EXP_FOLDER}/tmp_dir/test_rec
GT_DIR=${EXP_FOLDER}/tmp_dir/test_gt
ROOT=$PWD

echo "pred_folder" ${PRED_DIR}
echo "gt_folder" ${GT_DIR}

# FVD eval
# the root dir should be the dir containing the fvd pretrain model (resnet-50-kinetics and i3d)
python3  tool/metrics/metric_center.py \
--root_dir $ROOT --path_gen ${PRED_DIR} --path_gt ${GT_DIR} --type fid-vid fvd  \
 --write_metric_to ${EXP_FOLDER}/logs/metrics_fid-vid_fvd.json --number_sample_frames 16 --sample_duration 16

# L1 SSIM LPIPS and PSNR
# python3  tool/metrics/metric_center.py \
# --root_dir $ROOT --path_gen ${PRED_DIR}/ --path_gt ${GT_DIR} \
# --type ssim lpips psnr --write_metric_to ${EXP_FOLDER}/logs/metrics_l1_ssim_lpips_psnr.json
# ==============================================

rm -rf experiments/${EXP_NAME}/tmp_dir
