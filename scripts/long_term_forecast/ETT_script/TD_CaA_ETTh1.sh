#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

model_name=TD_CaA
seq_len=96
pred_lens=(96)

for pred_len in "${pred_lens[@]}"; do
    python run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_${seq_len}_${pred_len} \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_ff 128 \
      --des 'Exp' \
      --d_model 32 \
      --dropout 0.2 \
      --moving_avg 10 \
      --batch_size 8 \
      --learning_rate 0.00025 \
      --train_epochs 20 \
      --k_lookback 96 \
      --patience 10 \
      --fft \
      --itr 1
done