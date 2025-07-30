

export BATCH_SIZE=256
export HIDDEN_SIZE=2048
export N_BLOCKS=3
export LOGGER_NAME='01_2048_3_max1e3'
export MAX_LR=1e-3

# stage 1
docker run --rm --shm-size=5gb --gpus '"device=0"' \
    -v "$PWD/configs:/app/quarc/configs" \
    -v "$PWD/checkpoints:/app/quarc/checkpoints" \
    -v "$PWD/data:/app/quarc/data" \
    -v "$PROCESSED_DATA_DIR:/app/quarc/data/processed" \
    -t ${ASKCOS_REGISTRY}/quarc:1.0-gpu \
python quarc_trainer.py \
    --model-type ffn \
    --stage 1 \
    --logger-name $LOGGER_NAME \
    --save-dir ./retrained_models \
    --max-epochs 30 \
    --batch-size $((BATCH_SIZE / NUM_GPUS)) \
    --max-lr $MAX_LR \
    --hidden-size $HIDDEN_SIZE \
    --n-blocks $N_BLOCKS \
    --output-size 1376 \
    --num-classes 1376 \
    --processed-data-dir ./data/processed \
    --num-workers 8 \
    --train-data-path /home/xiaoqis/projects/cond_rec_clean/data/processed/stage1/stage1_train.pickle \
    --val-data-path /home/xiaoqis/projects/cond_rec_clean/data/processed/stage1/stage1_val.pickle \

# stage 2
