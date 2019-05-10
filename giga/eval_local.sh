#!/usr/bin/env bash

set -x
set -e

if [ $# -lt 3 ]; then
    echo 'Usage: ./eval_local.sh test.trg prediction.trg eval.log'
    exit 1
fi

test_trg=$1
predict_trg=$2
eval_log=$3
tmp_dir=./TGjwDjQ/

python3 ./giga/calc_rouge.py --test-trg ${test_trg} --predict-trg ${predict_trg} --tmp-dir ${tmp_dir} --eval-log ${eval_log}

rm -rf ${tmp_dir}

echo 'Done'