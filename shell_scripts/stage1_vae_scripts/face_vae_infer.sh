# =================
# Record scripts for vae training and inference
# =================

DATA_ROOT=datasets/Faceforensic++
DATA_IMG=configs/faceforensic/data/facevidimg.yaml
DATA_FLATTEN=configs/faceforensic/data/facevidflatten.yaml
MODEL=configs/faceforensic/vae/vae_4ch_32len_large.yaml

EXP_NAME=faceforensics_vae

bash shell_scripts/base_vae/infer_metric_script.sh $EXP_NAME $DATA_FLATTEN $MODEL
# bash shell_scripts/base_vae/infer_savelatent_script.sh $EXP_NAME $DATA_FLATTEN $MODEL $DATA_ROOT
