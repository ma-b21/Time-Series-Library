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
      --data_path ETTh1.csv \
      --model_id ETTh1_96_96 \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
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
      --d_model 64 \
      --dropout 0.25 \
      --moving_avg 10 \
      --batch_size 4 \
      --learning_rate 0.0015 \
      --train_epochs 20 \
      --k_lookback 48 \
      --patience 3 \
      --fft \
      --itr 1\
      --method "Dynamic"\
      --hidden 24\
      --bias \
      --interact
done

# for pred_len in "${pred_lens[@]}"; do
#     python run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTh1.csv \
#       --model_id ETTh1_96_192 \
#       --model $model_name \
#       --data ETTh1 \
#       --features M \
#       --seq_len 96 \
#       --label_len 48 \
#       --pred_len 192 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --d_ff 128 \
#       --des 'Exp' \
#       --d_model 14 \
#       --dropout 0.25 \
#       --moving_avg 10 \
#       --batch_size 4 \
#       --learning_rate 0.001 \
#       --train_epochs 20 \
#       --k_lookback 96 \
#       --patience 3 \
#       --fft \
#       --itr 1\
#       --method "Dynamic"\
#       --hidden 24\
#       --bias \
#       --interact
# done

# for pred_len in "${pred_lens[@]}"; do
#     python run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTh1.csv \
#       --model_id ETTh1_96_336 \
#       --model $model_name \
#       --data ETTh1 \
#       --features M \
#       --seq_len 96 \
#       --label_len 48 \
#       --pred_len 336 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --d_ff 128 \
#       --des 'Exp' \
#       --d_model 14 \
#       --dropout 0.15 \
#       --moving_avg 10 \
#       --batch_size 3 \
#       --learning_rate 0.001 \
#       --train_epochs 20 \
#       --k_lookback 96 \
#       --patience 3 \
#       --fft \
#       --itr 1\
#       --method "Dynamic"\
#       --hidden 24\
#       --bias \
#       --interact
# done

# for pred_len in "${pred_lens[@]}"; do
#     python run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTh1.csv \
#       --model_id ETTh1_96_720 \
#       --model $model_name \
#       --data ETTh1 \
#       --features M \
#       --seq_len 96 \
#       --label_len 48 \
#       --pred_len 720 \
#       --e_layers 2 \
#       --d_layers 1 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --d_ff 128 \
#       --des 'Exp' \
#       --d_model 14 \
#       --dropout 0.15 \
#       --moving_avg 10 \
#       --batch_size 2 \
#       --learning_rate 0.0015 \
#       --train_epochs 20 \
#       --k_lookback 96 \
#       --patience 3 \
#       --fft \
#       --itr 1\
#       --method "Dynamic"\
#       --hidden 24\
#       --bias \
#       --interact
# done