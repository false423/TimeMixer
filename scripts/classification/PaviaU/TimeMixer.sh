export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

e_layers=4
down_sampling_layers=1
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=32
batch_size=16


python -m debugpy --listen 888 --wait-for-client run.py \
  --task_name classification \
  --is_training 1 \
  --root_path 'D:\DeepLearning\Time\dataset\Pavia' \
  --data_path PaviaU.csv \
  --model_id PaviaU \
  --model $model_name \
  --data PaviaU \
  --features MS \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 128 \
  --d_model $d_model \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $learning_rate \
  --train_epochs 10 \
  --patience 20 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --loss 'SMAPE' \
  --use_gpu True \
  --pred_len 0 \
  --decomp_method 'dft_decomp' \
  --device 0 
