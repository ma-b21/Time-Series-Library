export CUDA_VISIBLE_DEVICES=2

model_name=TD_CaA
des='TD_CaA-MS'

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_96 \
#   --model $model_name \
#   --data ETTh1 \
#   --features MS \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_ff 512 \
#   --des $des \
#   --dropout 0.2 \
#   --d_model 14 \
#   --k_lookback 96 \
#   --batch_size 32 \
#   --moving_avg 1 \
#   --learning_rate 0.0003\
#   --train_epochs 20 \
#   --patience 3 \
#   --fft \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_192 \
#   --model $model_name \
#   --data ETTh1 \
#   --features MS \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 14 \
#   --d_ff 128 \
#   --batch_size 4 \
#   --des $des \
  # --dropout 0.1 \
  # --d_model 28 \
  # --k_lookback 96 \
  # --batch_size 32 \
  # --moving_avg 1 \
  # --learning_rate 0.001\
  # --train_epochs 20 \
  # --patience 3 \
  # --fft \
  # --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_336 \
#   --model $model_name \
#   --data ETTh1 \
#   --features MS \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_ff 512 \
#   --batch_size 32 \
#   --des $des \
#   --dropout 0.2 \
#   --d_model  128\
#   --k_lookback 96 \
#   --batch_size 32 \
#   --moving_avg 1 \
#   --learning_rate 0.0005\
#   --train_epochs 20 \
#   --patience 3 \
#   --fft \
#   --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --batch_size 128 \
  --des $des \
  --dropout 0.2 \
  --d_model  256\
  --k_lookback 96 \
  --batch_size 32 \
  --moving_avg 1 \
  --learning_rate 0.0005\
  --train_epochs 20 \
  --patience 3 \
  --fft \
  --itr 1
