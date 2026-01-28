#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=TD_CaA
seq_len=96
pred_lens=(96)

# for pred_len in "${pred_lens[@]}"; do
#     python run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTm2.csv \
#       --model_id ETTm2_${seq_len}_96 \
#       --model $model_name \
#       --data ETTm2 \
#       --features M \
#       --seq_len $seq_len \
#       --label_len 48 \
#       --pred_len 96 \
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
#       --dropout 0.15 \
#       --batch_size 32 \
#       --learning_rate 0.05 \
#       --train_epochs 20 \
#       --k_lookback 96 \
#       --patience 3 \
#       --method "Dynamic"\
#       --hidden 28 \
#       --bias \
#       --interact
# done

for pred_len in "${pred_lens[@]}"; do
    python run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm2.csv \
      --model_id ETTm2_${seq_len}_192 \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len 192 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_ff 128 \
      --des 'Exp' \
      --itr 1 \
      --d_model 64 \
      --dropout 0.05 \
      --batch_size 32 \
      --learning_rate 0.05 \
      --train_epochs 20 \
      --k_lookback 96 \
      --patience 3 \
      --method "Dynamic"\
      --hidden 26 \
      --bias \
      --interact
done

# for pred_len in "${pred_lens[@]}"; do
#     python run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTm2.csv \
#       --model_id ETTm2_${seq_len}_336 \
#       --model $model_name \
#       --data ETTm2 \
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
#       --d_model 64 \
#       --dropout 0.15 \
#       --batch_size 32 \
#       --learning_rate 0.05 \
#       --train_epochs 20 \
#       --k_lookback 96 \
#       --patience 3 \
#       --method "Dynamic"\
#       --hidden 28 \
#       --bias \
#       --interact
# done

# for pred_len in "${pred_lens[@]}"; do
#     python run.py \
#       --task_name long_term_forecast \
#       --is_training 1 \
#       --root_path ./dataset/ETT-small/ \
#       --data_path ETTm2.csv \
#       --model_id ETTm2_${seq_len}_720 \
#       --model $model_name \
#       --data ETTm2 \
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
#       --dropout 0.15 \
#       --batch_size 32 \
#       --learning_rate 0.05 \
#       --train_epochs 20 \
#       --k_lookback 96 \
#       --patience 3 \
#       --method "Dynamic"\
#       --hidden 28 \
#       --bias \
#       --interact
# done