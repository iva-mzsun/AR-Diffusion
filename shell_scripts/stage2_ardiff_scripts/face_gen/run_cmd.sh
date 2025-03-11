# Diffusion Forcing
# DATA_FILE_NAME=facelatent
# MODEL_FILE_NAME=diff_forcing
# bash shell_scripts/stage2_ardiff_scripts/face_gen/train_base_script.sh ${DATA_FILE_NAME} ${MODEL_FILE_NAME}

# AR Diffusion
DATA_FILE_NAME=facelatent
MODEL_FILE_NAME=diff_ardiff_vtattn_x0pred_nvae_midt
bash shell_scripts/stage2_ardiff_scripts/face_gen/train_base_script.sh ${DATA_FILE_NAME} ${MODEL_FILE_NAME}

VALIDDATA_FILE=facevidflatten
TRAINDATA_FILE=$DATA_FILE_NAME
bash shell_scripts/stage2_ardiff_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 16 5
# bash shell_scripts/stage2_ardiff_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 16 0
# bash shell_scripts/stage2_ardiff_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 1.0 128 5
