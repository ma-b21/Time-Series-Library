#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

model_name=TD_CaA
seq_len=96

python run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_${seq_len}_96 \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len 96 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_ff 128 \
      --des 'Exp' \
      --itr 1 \
      --d_model 96 \
      --dropout 0.05 \
      --batch_size 32 \
      --learning_rate 0.05 \
      --k_lookback 96 \
      --train_epochs 20 \
      --patience 3 \
      --method "Dynamic"\
      --hidden 28 \
      --bias \
      --interact


# python run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTh2.csv \
#       --model_id ETTh2_${seq_len}_192 \
#       --model $model_name \
#       --data ETTh2 \
#       --features M \
#       --seq_len $seq_len \
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
#       --itr 1 \
#       --d_model 64 \
#       --dropout 0.05 \
#       --batch_size 32 \
#       --moving_avg 3 \
#       --learning_rate 0.0475 \
#       --k_lookback 48 \
#       --train_epochs 20 \
#       --patience 3 \
#       --method "Dynamic"\
#       --hidden 26 \
#       --bias \
#       --interact


# python run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTh2.csv \
#       --model_id ETTh2_${seq_len}_336 \
#       --model $model_name \
#       --data ETTh2 \
#       --features M \
#       --seq_len $seq_len \
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
#       --itr 1 \
#       --d_model 32 \
#       --dropout 0.25 \
#       --batch_size 32 \
#       --moving_avg 3 \
#       --learning_rate 0.03 \
#       --k_lookback 32 \
#       --train_epochs 20 \
#       --patience 3 \
#       --method "Dynamic"\
#       --hidden 16 \
#       --bias \
#       --interact


# python run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTh2.csv \
#       --model_id ETTh2_${seq_len}_720 \
#       --model $model_name \
#       --data ETTh2 \
#       --features M \
#       --seq_len $seq_len \
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
#       --itr 1 \
#       --d_model 64 \
#       --dropout 0.1 \
#       --batch_size 32 \
#       --moving_avg 3 \
#       --learning_rate 0.00375 \
#       --k_lookback 96 \
#       --train_epochs 20 \
#       --patience 3 \
#       --method "Dynamic"\
#       --hidden 26 \
#       --bias \
#       --interact
