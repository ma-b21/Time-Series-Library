export CUDA_VISIBLE_DEVICES=3

model_name=TD_CaA
des='TD_CaA-MS'


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_96 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des $des \
#   --dropout 0.05 \
#   --d_model 512 \
#   --k_lookback 8 \
#   --batch_size 32 \
#   --moving_avg 1 \
#   --learning_rate 0.0001 \
#   --train_epochs 20 \
#   --patience 3 \
#   --fft \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_192 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des $des \
#   --dropout 0.05 \
#   --d_model 512 \
#   --k_lookback 4 \
#   --batch_size 32 \
#   --moving_avg 1 \
#   --learning_rate 0.0001 \
#   --train_epochs 20 \
#   --patience 3 \
#   --fft \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_336 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des $des \
#   --dropout 0.05 \
#   --d_model 512 \
#   --k_lookback 4 \
#   --batch_size 32 \
#   --moving_avg 1 \
#   --learning_rate 0.0001 \
#   --train_epochs 20 \
#   --patience 3 \
#   --fft \
#   --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des $des \
  --dropout 0.05 \
  --d_model 512 \
  --k_lookback 4 \
  --batch_size 32 \
  --moving_avg 1 \
  --learning_rate 0.0003 \
  --train_epochs 20 \
  --patience 3 \
  --fft \
  --itr 1