#!/usr/bin/env bash

node=$(hostname -s)

echo "Please Connecto to ${node}"

bert-serving-start -model_dir ./bert_models -num_worker=1 -max_seq_len=125
