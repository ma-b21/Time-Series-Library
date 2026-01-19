export CUDA_VISIBLE_DEVICES=2

model_name=TD_CaA

python -u run.py \
  # --task_name long_term_forecast \
  # --is_training 1 \
  # --root_path ./dataset/futures/ \
  # --data_path futures_CF.csv \
  # --model_id Futures_96_96 \
  # --model $model_name \
  # --data Futures \
  # --target LastPrice \
  # --features MS \
  # --seq_len 96 \
  # --label_len 48 \
  # --pred_len 96 \
  # --e_layers 1 \
  # --factor 3 \
  # --enc_in 13 \
  # --dec_in 13 \
  # --c_out 7 \
  # --dropout 0.1 \
  # --d_model 128 \
  # --k_lookback 8 \
  # --batch_size 32 \
  # --moving_avg 1 \
  # --learning_rate 0.0001 \
  # --train_epochs 20 \
  # --patience 10 \
  # --fft \
  # --des 'exp' \
  # --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/futures/ \
  --data_path futures_CF.csv \
  --model_id Futures_96_96 \
  --model $model_name \
  --data Futures \
  --target LastPrice \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --dropout 0.1 \
  --d_model 128 \
  --k_lookback 8 \
  --batch_size 32 \
  --moving_avg 1 \
  --learning_rate 0.000 \
  --train_epochs 20 \
  --patience 10 \
  --fft \
  --des 'exp' \
  --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_336 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 1024 \
#   --batch_size 16 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_720 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 1024 \
#   --batch_size 16 \
#   --itr 1
