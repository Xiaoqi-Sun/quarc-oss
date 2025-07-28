
# hparams
HIDDEN_SIZE=2048
N_BLOCKS=3
LOGGER_NAME='G01_2048_3_max1e3_d2_gh1024'
MAX_LR=1e-3
BATCH_SIZE=512
NUM_GPUS=2 # change

GRAPH_HIDDEN_SIZE=1024
DEPTH=2

# ffn train (total batch size = 512)
torchrun --nproc_per_node=2 quarc_trainer.py \
    --model-type gnn \
    --stage 1 \
    --logger-name $LOGGER_NAME \
    --save-dir ./retrained_models \
    --max-epochs 30 \
    --batch-size $((BATCH_SIZE / NUM_GPUS)) \
    --max-lr $MAX_LR \
    --graph-hidden-size $GRAPH_HIDDEN_SIZE \
    --depth $DEPTH \
    --hidden-size $HIDDEN_SIZE \
    --n-blocks $N_BLOCKS \
    --output-size 1376 \
    --num-classes 1376 \
    --processed-data-dir ./data/processed \
    --num-workers 16 \
    --train-data-path /home/xiaoqis/projects/cond_rec_clean/data/processed/stage1/stage1_train.pickle \
    --val-data-path /home/xiaoqis/projects/cond_rec_clean/data/processed/stage1/stage1_val.pickle \

    # --early-stop
    # --early-stop-patience 5
    # --train-data-path /home/xiaoqis/projects/cond_rec_clean/data/processed/stage1/stage1_test_2000.pickle \
    # --val-data-path /home/xiaoqis/projects/cond_rec_clean/data/processed/stage1/stage1_test_2000.pickle \
