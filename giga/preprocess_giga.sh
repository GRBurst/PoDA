#!/usr/bin/env bash

set -x
set -e
source ./giga/paths.sh

mkdir -p $processed_dir

dict_size=50000

python3 -u ${SOFTWARE_DIR}/preprocess.py \
--source-lang src --target-lang trg \
--trainpref $processed_dir/train \
--validpref $processed_dir/valid \
--testpref $processed_dir/test \
--nwordssrc $dict_size \
--nwordstgt $dict_size \
--padding-factor 0 \
--srcdict ./da-pretrained/dict.src.txt \
--tgtdict ./da-pretrained/dict.trg.txt \
--destdir $processed_dir/bin
