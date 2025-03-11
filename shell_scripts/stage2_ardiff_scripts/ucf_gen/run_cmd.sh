# diffusion forcing
# DATA_FILE_NAME=ucflatent
# MODEL_FILE_NAME=diff_forcing
# bash shell_scripts/ucf_gen/train_base_script.sh ${DATA_FILE_NAME} ${MODEL_FILE_NAME}

# bash shell_scripts/ucf_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 16 0
# bash shell_scripts/ucf_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 128 5

# 无条件生成
DATA_FILE_NAME=ucflatent
MODEL_FILE_NAME=diff_ardiff_vtattn_x0pred_nvae_midt
# bash shell_scripts/ucf_gen/train_base_script.sh ${DATA_FILE_NAME} ${MODEL_FILE_NAME}
# PRETRAINED_CKPT=experiments/2024-10-25T18-01-38_diff_ardiff_vtattn_x0pred_nvae_midt_class_ucflatent/ckpt_dir/checkpoint-195000
# bash shell_scripts/ucf_gen/finetune_base_script.sh ${DATA_FILE_NAME} ${MODEL_FILE_NAME} ${PRETRAINED_CKPT}
VALIDDATA_FILE=ucfvidflatten
TRAINDATA_FILE=$DATA_FILE_NAME
bash shell_scripts/ucf_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 16 0
# bash shell_scripts/ucf_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 128 5
# bash shell_scripts/ucf_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 128 10
# bash shell_scripts/ucf_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 128 25
# bash shell_scripts/ucf_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 128 50


# 类别条件生成
# DATA_FILE_NAME=ucflatent
# MODEL_FILE_NAME=diff_ardiff_vtattn_x0pred_nvae_midt_class
# # bash shell_scripts/ucf_gen/debug_base_script.sh ${DATA_FILE_NAME} ${MODEL_FILE_NAME}
# bash shell_scripts/ucf_gen/train_base_script.sh ${DATA_FILE_NAME} ${MODEL_FILE_NAME}
