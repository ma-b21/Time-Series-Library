#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=EATA
des='EATA-MS'
seq_len=36
label_len=18
enc_in=7
dec_in=7
c_out=1
e_layers=4
factor=3
train_epochs=10
patience=3
k_lookback=36
hidden=28

for pred_len in 24 36 48 60; do
  case "$pred_len" in
    24)
      dropout=0.05
      batch_size=32
      d_model=64
      learning_rate=0.05
      ;;
    36)
      dropout=0.15
      batch_size=16
      d_model=64
      learning_rate=0.01
      ;;
    48)
      dropout=0.25
      batch_size=12
      d_model=64
      learning_rate=0.005
      ;;
    60)
      dropout=0.15
      batch_size=16
      d_model=64
      learning_rate=0.001
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
    --d_layers 1 \
    --factor $factor \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --c_out $c_out \
    --des $des \
    --dropout $dropout \
    --batch_size $batch_size \
    --d_model $d_model \
    --k_lookback $k_lookback \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --method Dynamic \
    --hidden $hidden \
    --bias \
    --interact \
    --itr 1
done
