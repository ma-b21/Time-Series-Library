export CUDA_VISIBLE_DEVICES=0

model_name=TD_CaA
des='TD_CaA-MS'

# python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_96_96 \
#     --model $model_name \
#     --data ETTh1 \
#     --features MS \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --e_layers 2 \
#     --factor 3 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --d_ff 512 \
#     --des $des \
#     --itr 1 \
#     --dropout 0.2 \
#     --d_model 64 \
#     --k_lookback 96 \
#     --batch_size 32 \
#     --learning_rate 0.065 \
#     --train_epochs 20 \
#     --patience 3 \
#     --method "Dynamic"\
#     --hidden 26 \
#     --bias \
#     --interact

# python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_96_192 \
#     --model $model_name \
#     --data ETTh1 \
#     --features MS \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 192 \
#     --e_layers 2 \
#     --factor 3 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --d_ff 512 \
#     --des $des \
#     --itr 1 \
#     --dropout 0.2 \
#     --d_model 64 \
#     --k_lookback 96 \
#     --batch_size 32 \
#     --learning_rate 0.02 \
#     --train_epochs 20 \
#     --patience 3 \
#     --method "Dynamic"\
#     --hidden 28 \
#     --bias \
#     --interact

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_336 \
    --model $model_name \
    --data ETTh1 \
    --features MS \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 336 \
    --e_layers 2 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_ff 512 \
    --des $des \
    --itr 1 \
    --dropout 0.2 \
    --d_model 64 \
    --k_lookback 96 \
    --batch_size 32 \
    --learning_rate 0.03 \
    --train_epochs 20 \
    --patience 3 \
    --method "Dynamic"\
    --hidden 32 \
    --bias \
    --interact