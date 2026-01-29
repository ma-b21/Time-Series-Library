#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

model_name=TD_CaA
seq_len=96
    
# python run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTm1.csv \
#       --model_id ETTm1_${seq_len}_96 \
#       --model $model_name \
#       --data ETTm1 \
#       --features M \
#       --seq_len $seq_len \
#       --label_len 48 \
#       --pred_len 96 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --d_ff 128 \
#       --des 'Exp' \
#       --itr 1 \
#       --d_model 128 \
#       --dropout 0.05 \
#       --batch_size 32 \
#       --learning_rate 0.0008 \
#       --train_epochs 15 \
#       --k_lookback 96 \
#       --patience 3 \
#       --method "Dynamic"\
#       --hidden 64 \
#       --bias \
#       --interact


# python run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTm1.csv \
#       --model_id ETTm1_${seq_len}_192 \
#       --model $model_name \
#       --data ETTm1 \
#       --features M \
#       --seq_len $seq_len \
#       --label_len 48 \
#       --pred_len 192 \
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
#       --learning_rate 0.00025 \
#       --train_epochs 15 \
#       --k_lookback 96 \
#       --patience 3 \
#       --method "Dynamic"\
#       --hidden 34 \
#       --bias \
#       --interact


# python run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTm1.csv \
#       --model_id ETTm1_${seq_len}_336 \
#       --model $model_name \
#       --data ETTm1 \
#       --features M \
#       --seq_len $seq_len \
#       --label_len 48 \
#       --pred_len 336 \
#       --factor 3 \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --d_ff 128 \
#       --des 'Exp' \
#       --itr 1 \
#       --d_model 64 \
#       --dropout 0.1 \
#       --batch_size  32 \
#       --learning_rate 0.0002 \
#       --train_epochs 15 \
#       --k_lookback 96 \
#       --patience 3 \
#       --method "Dynamic"\
#       --hidden 28 \
#       --bias \
#       --interact

python run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm1.csv \
      --model_id ETTm1_${seq_len}_720 \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len 720 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_ff 128 \
      --des 'Exp' \
      --itr 1 \
      --d_model 64 \
      --dropout 0.05 \
      --batch_size  64 \
      --learning_rate 0.0005 \
      --train_epochs 15 \
      --k_lookback 96 \
      --patience 3 \
      --method "Dynamic"\
      --hidden 28 \
      --bias \
      --interact