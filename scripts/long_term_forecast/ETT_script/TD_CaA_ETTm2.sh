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
      --data_path ETTm2.csv \
      --model_id ETTm2_${seq_len}_${pred_len} \
      --model $model_name \
      --data ETTm2 \
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
      --itr 1 \
      --d_model 128 \
      --dropout 0.1 \
      --batch_size 32 \
      --learning_rate 0.00015 \
      --moving_avg 1 \
      --train_epochs 20 \
      --k_lookback 96 \
      --patience 3 \
      --fft 
done