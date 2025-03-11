export MODE=$1  # ['vae', 'diff']
export EXP_NAME=$2
export SAVE_DIR=$3

EXP_FOLDER=experiments/${EXP_NAME}
GT_DIR=${EXP_FOLDER}/$SAVE_DIR/test_gt

echo $MODE
if [ "$MODE" == 'vae' ] || [ "$MODE" == 'VAE' ]; then
    PRED_DIR=${EXP_FOLDER}/$SAVE_DIR/test_rec
elif [ "$MODE" == 'diff' ] || [ "$MODE" == 'DIFF' ]; then
    PRED_DIR=${EXP_FOLDER}/$SAVE_DIR/test_pred
else
    echo "None implementation!"; exit 0
fi

echo "pred_folder" ${PRED_DIR}
echo "gt_folder" ${GT_DIR}

if [ "$MODE" == 'vae' ] || [ "$MODE" == 'VAE' ]; then
    # L1 SSIM LPIPS and PSNR
    python3  tool/metrics/metric_center.py --path_gen ${PRED_DIR}/ --path_gt ${GT_DIR} --type ssim lpips psnr --write_metric_to ${EXP_FOLDER}/logs/metrics_l1_ssim_lpips_psnr.json
    # Pytorch-FID
    python3 -m pytorch_fid ${PRED_DIR} ${GT_DIR} --device cuda:0 > ${EXP_FOLDER}/logs/pytorch_fid.txt
fi

#  FVD eval
# the root dir should be the dir containing the fvd pretrain model (resnet-50-kinetics and i3d)
python3  tool/metrics/metric_center.py --path_gen ${PRED_DIR} --path_gt ${GT_DIR} --type fid-img fid-vid fvd  \
 --write_metric_to ${EXP_FOLDER}/logs/metrics_${MODE}_${SAVE_DIR}.json --number_sample_frames 16 --sample_duration 16
