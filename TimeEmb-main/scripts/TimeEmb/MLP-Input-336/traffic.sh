model_name=TimeEmb

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom
random_seed=2024
seq_len=336
for pred_len in 96 192 336 720
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --use_hour_index 1 \
      --hour_length 24 \
      --use_day_index 1 \
      --day_length 7 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 64 --learning_rate 0.002 --random_seed $random_seed
done
