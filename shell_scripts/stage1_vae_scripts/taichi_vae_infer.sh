# =================
# Record scripts for vae training and inference
# =================

DATA_ROOT=datasets/taichi
DATA_IMG=configs/taichi/data/vidimg.yaml
DATA_FLATTEN=configs/taichi/data/vidflatten.yaml
MODEL=configs/taichi/vae/vae_4ch_32len_large.yaml

EXP_NAME=taichi_vae

bash shell_scripts/base_vae/infer_metric_script.sh $EXP_NAME $DATA_FLATTEN $MODEL
# bash shell_scripts/base_vae/infer_savelatent_script.sh $EXP_NAME $DATA_FLATTEN $MODEL $DATA_ROOT
