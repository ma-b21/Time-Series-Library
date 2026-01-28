export CUDA_VISIBLE_DEVICES=0

model_name=TD_CaA

# python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/futures/ \
#     --data_path futures_TA.csv \
#     --model_id Futures_TA_96_96 \
#     --model $model_name \
#     --data Futures \
#     --target LastPrice \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --e_layers 1 \
#     --factor 3 \
#     --enc_in 12 \
#     --dec_in 12 \
#     --c_out 12 \
#     --d_model 64 \
#     --dropout 0.2 \
#     --moving_avg 10 \
#     --batch_size 32 \
#     --learning_rate 0.005 \
#     --train_epochs 20 \
#     --k_lookback 96 \
#     --patience 3 \
#     --itr 1\
#     --method "Dynamic"\
#     --hidden 28 \
#     --bias \
#     --interact


# python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/futures/ \
#     --data_path futures_TA.csv \
#     --model_id Futures_TA_96_192 \
#     --model $model_name \
#     --data Futures \
#     --target LastPrice \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 192 \
#     --e_layers 1 \
#     --factor 3 \
#     --enc_in 12 \
#     --dec_in 12 \
#     --c_out 12 \
#     --d_model 32 \
#     --dropout 0.05 \
#     --moving_avg 10 \
#     --batch_size 32 \
#     --learning_rate 0.00075 \
#     --train_epochs 20 \
#     --k_lookback 96 \
#     --patience 3 \
#     --itr 1\
#     --method "Dynamic"\
#     --hidden 28 \
#     --bias \
#     --interact

# python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/futures/ \
#     --data_path futures_TA.csv \
#     --model_id Futures_TA_96_336 \
#     --model $model_name \
#     --data Futures \
#     --target LastPrice \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 336 \
#     --e_layers 1 \
#     --factor 3 \
#     --enc_in 12 \
#     --dec_in 12 \
#     --c_out 12 \
#     --d_model 64 \
#     --dropout 0.05 \
#     --moving_avg 10 \
#     --batch_size 32 \
#     --learning_rate 0.001 \
#     --train_epochs 20 \
#     --k_lookback 96 \
#     --patience 3 \
#     --itr 1\
#     --method "Dynamic"\
#     --hidden 28 \
#     --bias \
#     --interact

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/futures/ \
    --data_path futures_TA.csv \
    --model_id Futures_TA_96_720 \
    --model $model_name \
    --data Futures \
    --target LastPrice \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --e_layers 1 \
    --factor 3 \
    --enc_in 12 \
    --dec_in 12 \
    --c_out 12 \
    --d_model 64 \
    --dropout 0.05 \
    --moving_avg 10 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --k_lookback 96 \
    --patience 3 \
    --itr 1\
    --method "Dynamic"\
    --hidden 28 \
    --bias \
    --interact