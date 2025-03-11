# =================
# Record scripts for vae training and inference
# =================

DATA_ROOT=datasets/ucf101
DATA_IMG=configs/ucf101/data/ucfvidimg.yaml
DATA_FLATTEN=configs/ucf101/data/ucfvidflatten.yaml
MODEL=configs/ucf101/vae/vae_4ch_32len_large.yaml

EXP_NAME=ucf101_vae

bash shell_scripts/base_vae/infer_metric_script.sh $EXP_NAME $DATA_FLATTEN $MODEL
# bash shell_scripts/base_vae/infer_savelatent_script.sh $EXP_NAME $DATA_FLATTEN $MODEL $DATA_ROOT
