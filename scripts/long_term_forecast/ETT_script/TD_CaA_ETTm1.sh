#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=TD_CaA
seq_len=96
pred_lens=(96)

for pred_len in "${pred_lens[@]}"; do
    python run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm1.csv \
      --model_id ETTm1_${seq_len}_${pred_len} \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_ff 128 \
      --des 'Exp' \
      --itr 1 \
      --d_model 64 \
      --dropout 0.2 \
      --batch_size  3\
      --learning_rate 0.0015 \
      --train_epochs 15 \
      --k_lookback 48 \
      --patience 3 \
      --method "Dynamic"\
      --hidden 15\
      --bias \
      --interact
done