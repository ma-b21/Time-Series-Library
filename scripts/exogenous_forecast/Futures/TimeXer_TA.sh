#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=TimeXer
des='TimeXer-MS'
data_path=futures_TA.csv
model_prefix=futures_TA
seq_len=96
label_len=48
enc_in=146
dec_in=146
c_out=1

for pred_len in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/futures/ \
    --data_path $data_path \
    --model_id ${model_prefix}_${seq_len}_${pred_len} \
    --model $model_name \
    --data Futures \
    --target LastPrice \
    --features MS \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --factor 3 \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --c_out $c_out \
    --d_model 128 \
    --d_ff 512 \
    --batch_size 32 \
    --des $des \
    --itr 1
done
