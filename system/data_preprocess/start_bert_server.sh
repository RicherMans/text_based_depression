#!/usr/bin/env bash

model_dir=${1:-"./bert_models"}
node=$(hostname -s)

echo "Please Connect to ${node}"

bert-serving-start -model_dir ${model_dir} -num_worker=4 -max_seq_len=125 -cpu
