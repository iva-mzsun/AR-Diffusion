# AR-Diffusion: Asynchronous Video Generation with Auto-Regressive Diffusion
![image](https://code.byted.org/sunmingzhen.triz/3d_tokenizer/blob/release/fig1.png)
![image](https://code.byted.org/sunmingzhen.triz/3d_tokenizer/blob/release/fig2.png)

# Environment
python 3.9
```
pip install -r requirements.txt
```

# Infer Scripts
Ensure all model checkpoints are put in the `experiments` folder.

1. Testing the reconstruction performance of VAE
```
bash shell_scripts/stage1_vae_scripts/face_vae_infer.sh
```

2. Testing the generation performance of AR Diffusion
```
DATA_FILE_NAME=facelatent
MODEL_FILE_NAME=diff_ardiff_vtattn_x0pred_nvae_midt
VALIDDATA_FILE=facevidflatten
TRAINDATA_FILE=$DATA_FILE_NAME
bash shell_scripts/stage2_ardiff_scripts/face_gen/infer_base_script.sh $TRAINDATA_FILE $VALIDDATA_FILE $MODEL_FILE_NAME 2.0 16 5
```

# Train Scripts

1. Training VAE on video frames.

【NOTE: Please download the tokenizer_titok_l32.bin file from https://huggingface.co/TrizZZZ/ar_diffusion and put it in the root folder before training the VAE.】
```
bash shell_scripts/stage1_vae_scripts/sky_vae_train.sh
```

2. Finetuning VAE on videos with temporal causal attention.
```
bash shell_scripts/stage1_vae_scripts/sky_vae_train_ftwt.sh
```

3. Extract video latents using VAEs for speeding up the training of AR Diffusion model.
```
Open the line of 'bash shell_scripts/base_vae/infer_savelatent_script.sh' in 
shell_scripts/stage1_vae_scripts/sky_vae_infer.sh
```

4. Training AR Diffusion model.
```
DATA_FILE_NAME=skyvidlatent
MODEL_FILE_NAME=diff_ardiff_vtattn_x0pred_nvae_midt
bash shell_scripts/stage2_ardiff_scripts/sky_gen/train_base_script.sh ${DATA_FILE_NAME} ${MODEL_FILE_NAME}
```

# Checkpoints
Checkpoints of VAE and AR-Diffusion models on the Sky-Timelapse, TaiChi-HD, UCF101, and Faceforensics datasets have been uploaded to the Huggingface hub: https://huggingface.co/TrizZZZ/ar_diffusion

# More Samples
More video samples can be viewed in: https://anonymouss765.github.io/AR-Diffusion

# Citation
TODO
```
@article{ardiff,
  title={AR-Diffusion: Asynchronous Video Generation with Auto-Regressive Diffusion},
  author={Sun, Mingzhen and Wang, Weining and Li, Gen and Liu, Jiawei and Sun, Jiahui and Feng, Wanquan and Lao, shanshan and Zhou, SiYu and He, Qian and Liu, Jing},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2025}
}
```
