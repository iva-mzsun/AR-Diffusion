# =================
# Record scripts for vae training and inference
# =================
DATA_ROOT=datasets/skytimelapse
DATA_IMG=configs/sky/data/skyvid_img.yaml
DATA_FLATTEN=configs/sky/data/skyvidflatten.yaml
MODEL=configs/sky/vae/vae_4ch_32len_large.yaml

EXP_NAME=skytimelapse_vae

bash shell_scripts/base_vae/infer_metric_script.sh $EXP_NAME $DATA_FLATTEN $MODEL
# bash shell_scripts/base_vae/infer_savelatent_script.sh $EXP_NAME $DATA_FLATTEN $MODEL $DATA_ROOT
