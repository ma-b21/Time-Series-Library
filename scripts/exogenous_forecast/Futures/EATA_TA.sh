#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=EATA
des='EATA-MS'
data_path=futures_TA.csv
model_prefix=Futures_TA
seq_len=96
label_len=48
enc_in=146
dec_in=146
c_out=1
e_layers=1
factor=3
train_epochs=20
patience=3
k_lookback=64
hidden=28

for pred_len in 96 192 336 720; do
  case "$pred_len" in
    96)
      dropout=0.2
      batch_size=32
      d_model=64
      learning_rate=0.005
      ;;
    192)
      dropout=0.05
      batch_size=32
      d_model=64
      learning_rate=0.00075
      ;;
    336)
      dropout=0.05
      batch_size=32
      d_model=64
      learning_rate=0.001
      ;;
    720)
      dropout=0.05
      batch_size=32
      d_model=64
      learning_rate=0.001
      ;;
  esac

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
    --e_layers $e_layers \
    --factor $factor \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --c_out $c_out \
    --des $des \
    --dropout $dropout \
    --d_model $d_model \
    --k_lookback $k_lookback \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --method Dynamic \
    --hidden $hidden \
    --bias \
    --interact \
    --itr 1
done
