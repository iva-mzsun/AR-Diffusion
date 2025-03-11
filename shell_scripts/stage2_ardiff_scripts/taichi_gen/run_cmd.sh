# DATA_FILE_NAME=taichilatent
# MODEL_FILE_NAME=diff_forcing
# bash shell_scripts/taichi_gen/train_base_script.sh ${DATA_FILE_NAME} ${MODEL_FILE_NAME}
# VALIDDATA_FILE=vidflatten
# TRAINDATA_FILE=$DATA_FILE_NAME
# bash shell_scripts/taichi_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 16 0
# bash shell_scripts/taichi_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 16 0
# bash shell_scripts/taichi_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 128 5

# ARDiff
DATA_FILE_NAME=taichilatent
MODEL_FILE_NAME=diff_ardiff_vtattn_x0pred_nvae_midt
# bash shell_scripts/taichi_gen/train_base_script.sh ${DATA_FILE_NAME} ${MODEL_FILE_NAME}
# PRETRAINED_CKPT=experiments/2024-10-24T14-01-30_diff_ardiff_vtattn_x0pred_nvae_midt_taichilatent/ckpt_dir/checkpoint-40000
# bash shell_scripts/taichi_gen/finetune_base_script.sh ${DATA_FILE_NAME} ${MODEL_FILE_NAME} ${PRETRAINED_CKPT}

VALIDDATA_FILE=vidflatten
TRAINDATA_FILE=$DATA_FILE_NAME
bash shell_scripts/taichi_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 16 0
bash shell_scripts/taichi_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 16 5
# bash shell_scripts/taichi_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 128 5
# bash shell_scripts/taichi_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 128 10
# bash shell_scripts/taichi_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 128 25
# bash shell_scripts/taichi_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 128 50
