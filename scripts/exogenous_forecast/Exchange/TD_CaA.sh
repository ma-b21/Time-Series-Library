export CUDA_VISIBLE_DEVICES=1

model_name=TD_CaA

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_ff 512 \
  --des 'Exp' \
  --dropout 0.05 \
  --d_model 128 \
  --k_lookback 4 \
  --batch_size 4 \
  --moving_avg 1 \
  --learning_rate 0.0001 \
  --train_epochs 20 \
  --patience 3 \
  --fft \
  --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_192 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 3 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --d_ff 512 \
#   --des 'Exp' \
#   --dropout 0.05 \
#   --d_model 256 \
#   --k_lookback 4 \
#   --batch_size 16 \
#   --moving_avg 1 \
#   --learning_rate 0.0001 \
#   --train_epochs 20 \
#   --patience 3 \
#   --fft \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_336 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --d_ff 512 \
#   --des 'Exp' \
#   --dropout 0.05 \
#   --d_model 128 \
#   --k_lookback 4 \
#   --batch_size 8 \
#   --moving_avg 1 \
#   --learning_rate 0.00005 \
#   --train_epochs 1 \
#   --patience 3 \
#   --fft \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_720 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --factor 3 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --d_ff 512 \
#   --des 'Exp' \
#   --dropout 0.05 \
#   --d_model 128 \
#   --k_lookback 4 \
#   --batch_size 8 \
#   --moving_avg 1 \
#   --learning_rate 0.00001 \
#   --train_epochs 2 \
#   --patience 3 \
#   --fft \
#   --ab_flag 1\
#   --itr 1
