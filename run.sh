#!/bin/bash

echo "Start to training on BERT-base"
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name_or_path bert-base-uncased \
    --c_model_name_or_path unsup-simcse-bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-simcse-bert-base-uncased \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model avg_sts \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --phi 0.9 \
    --noise_times 1 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --is_base True \
    #--fp16 \
    "$@"

CUDA_VISIBLE_DEVICES=0 python evaluation.py \
	      --model_name_or_path result/my-unsup-simcse-bert-base-uncased \
        --pooler cls_before_pooler \
	      --task_set sts \
        --mode test

echo "Start to training on RoBERTa-base"
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name_or_path roberta-base \
    --c_model_name_or_path unsup-simcse-roberta-base \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-simcse-roberta-base \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model avg_sts \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --phi 0.85 \
    --noise_times 2.5 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --is_base True \
    #--fp16 \
    "$@"

CUDA_VISIBLE_DEVICES=0 python evaluation.py \
	      --model_name_or_path result/my-unsup-simcse-roberta-base \
        --pooler cls_before_pooler \
	      --task_set sts \
        --mode test


echo "Start to training on BERT-large"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 55555 train.py \
    --model_name_or_path unsup-simcse-bert-large-uncased \
    --c_model_name_or_path unsup-simcse-bert-large-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-simcse-bert-large-uncased \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model avg_sts \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --phi 0.9 \
    --noise_times 3 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --is_base False \
    #--fp16 \
    "$@"

CUDA_VISIBLE_DEVICES=0 python evaluation.py \
	      --model_name_or_path result/my-unsup-simcse-bert-large-uncased \
        --pooler cls_before_pooler \
	      --task_set sts \
        --mode test

echo "Start to training on RoBERTa-large"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 55555 train.py \
    --model_name_or_path unsup-simcse-roberta-large \
    --c_model_name_or_path unsup-simcse-roberta-large \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-simcse-roberta-large \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model avg_sts \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --phi 0.85 \
    --noise_times 5 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --is_base False \
    #--fp16 \
    "$@"

CUDA_VISIBLE_DEVICES=0 python evaluation.py \
	      --model_name_or_path result/my-unsup-simcse-roberta-large \
        --pooler cls_before_pooler \
	      --task_set sts \
        --mode test