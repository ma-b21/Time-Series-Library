export CUDA_VISIBLE_DEVICES=0

model_name=TimeXer
des='Timexer-MS'


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/futures/ \
  --data_path futures_CF.csv \
  --model_id futures_CF_96_96 \
  --model $model_name \
  --data Futures \
  --target LastPrice \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 146 \
  --dec_in 146 \
  --c_out 1 \
  --des $des \
  --d_model 128 \
  --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/futures/ \
#   --data_path futures_CF.csv \
#   --model_id futures_CF_96_96 \
#   --model $model_name \
#   --data Futures \
#   --target LastPrice \
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
#   --d_model 128 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/futures/ \
#   --data_path futures_CF.csv \
#   --model_id futures_CF_96_96 \
#   --model $model_name \
#   --data Futures \
#   --target LastPrice \
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
#   --d_model 128 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/weather/ \
#   --data_path weather.csv \
#   --model_id weather_96_720 \
#   --model $model_name \
#   --data custom \
#   --features MS \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des $des \
#   --d_model 128 \
#   --itr 1