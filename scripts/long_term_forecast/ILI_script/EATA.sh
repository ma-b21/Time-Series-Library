export CUDA_VISIBLE_DEVICES=0

model_name=EATA

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_36_24 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 24 \
#   --e_layers 4 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --dropout 0.05 \
#   --batch_size 32 \
#   --d_model 64 \
#   --moving_avg 10 \
#   --k_lookback 36 \
#   --learning_rate 0.05 \
#   --train_epochs 10 \

  

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_36_36 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 36 \
#   --e_layers 4 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --dropout 0.15 \
#   --batch_size 16 \
#   --d_model 64 \
#   --k_lookback 36 \
#   --learning_rate 0.01 \
#   --train_epochs 10 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --dropout 0.25 \
  --batch_size 12 \
  --d_model 64 \
  --k_lookback 36 \
  --learning_rate 0.005 \
  --train_epochs 10 \


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_36_60 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 60 \
#   --e_layers 4 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_heads 16 \
#   --d_model 2048\
#   --itr 1