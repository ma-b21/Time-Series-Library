#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=TimeXer
des='TimeXer-MS'
seq_len=36
label_len=18
enc_in=7
dec_in=7
c_out=1
factor=3

for pred_len in 24 36 48 60; do
  case "$pred_len" in
    24)
      e_layers=1
      d_model=256
      d_ff=1024
      batch_size=16
      ;;
    36)
      e_layers=1
      d_model=256
      d_ff=1024
      batch_size=16
      ;;
    48)
      e_layers=2
      d_model=512
      d_ff=1024
      batch_size=4
      ;;
    60)
      e_layers=2
      d_model=256
      d_ff=1024
      batch_size=16
      ;;
  esac

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --model_id ili_${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --target OT \
    --features MS \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers $e_layers \
    --factor $factor \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --c_out $c_out \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --des $des \
    --itr 1
done
