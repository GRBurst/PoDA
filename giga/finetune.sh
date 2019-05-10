#!/usr/bin/env bash

set -x
set -e
source ./giga/paths.sh

device_id=0
seed=2145
model_name='giga'
da_pretrain='./da-pretrained/checkpoint_5M.pt'
if [ $# -ge 1 ]; then
    device_id=$1
fi
if [ $# -ge 2 ]; then
    model_name=$2
fi
if [ $# -ge 3 ]; then
    da_pretrain=$3
fi
if [ $# -ge 4 ]; then
    seed=$4
fi
if [ $# -ge 5 ]; then
    processed_dir=$5
fi


DATA_BIN_DIR=$processed_dir/bin

OUT_DIR=models_giga/model_$model_name/
mkdir -p $OUT_DIR

LOG_FILE=${OUT_DIR}log.out
PID_FILE=${OUT_DIR}pid.out

PYTHONPATH=$SOFTWARE_DIR:$PYTHONPATH CUDA_VISIBLE_DEVICES=$device_id nohup python3 -u $SOFTWARE_DIR/train.py  \
$DATA_BIN_DIR  \
--save-dir $OUT_DIR  \
--max-epoch 100  \
--batch-size 64  \
--max-tokens 3000 \
--no-progress-bar  \
--train-subset train:1 \
--valid-subset valid \
--seed $seed  \
--arch transformer  \
--warmup-updates 100 \
--clip-norm 2 --lr 1e-3 --min-lr 1e-5 --lr-shrink 0.5  \
--dropout 0.2 --relu-dropout 0.2 --attention-dropout 0.2 \
--encoder-embed-dim 512 --decoder-embed-dim 512 \
--encoder-word-embed-dim 512  --decoder-word-embed-dim 512 \
--max-target-positions 500 --max-source-positions 500  \
--encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 4096  \
--share-all-embeddings \
--use-copy True \
--log-interval 500 \
--da-pretrain-checkpoint $da_pretrain \
> $LOG_FILE 2>&1 &

echo $! > $PID_FILE
cat $PID_FILE
