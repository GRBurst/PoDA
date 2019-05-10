#!/usr/bin/env bash

set -e
set -x
source ./giga/paths.sh

if [ $# -ge 3 ]; then
    output_dir=$1
    device=$2
    model_path=$3
    sub_set='test'
    if [ $# -ge 4 ]; then
        sub_set=$4
    fi
else
    echo "Please specify the paths to the input_file and output directory"
    echo "Usage: `basename $0` <output_dir> <gpu-device-num(e.g: 0)> <path to model_file/dir>" >&2
    exit -1
fi

if [[ ! -e "$model_path" ]]; then
    echo "Model path not found: $model_path"
    exit -1
fi


beam=12

mkdir -p $output_dir

#PYTHONPATH  running fairseq on the test data
CUDA_VISIBLE_DEVICES=$device python -u $SOFTWARE_DIR/generate.py $processed_dir/bin \
--path $model_path  \
--beam $beam \
--nbest $beam  \
--gen-subset $sub_set  \
--max-sentences 64 \
--max-tokens 3000 \
--max-len-b 40 \
--no-progress-bar  \
--use-copy True \
--raw-text \
> $output_dir/output.nbest.txt

# getting best hypotheses
cat $output_dir/output.nbest.txt | grep "^H"  | python ./utils/sort.py $beam $output_dir/output.tok.txt
