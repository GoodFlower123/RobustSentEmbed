# RobustSentEmbed
a self-supervised sentence embedding framework that enhances both generalization and robustness benchmarks

### Train the RobustSentEmbed embeddings to generate robust text represnetation

#!/bin/bash

LR=7e-6
MASK=0.30
LAMBDA=0.005

#!python train.py \
!python -m torch.distributed.launch --nproc_per_node 4 --master_port $(expr $RANDOM + 1000) train.py \
    --model_name_or_path bert-base-uncased \
    --generator_name distilbert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir /data/long/jasl/RNLP/result/DiffSCE7_bert \
    --num_train_epochs 2 \
    --per_device_train_batch_size 64 \
    --learning_rate 7e-6 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model avg_sts \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --logging_first_step \
    --logging_dir your_logging_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --batchnorm \
    --lambda_weight 0.005 \
    --fp16 --masking_ratio 0.20

