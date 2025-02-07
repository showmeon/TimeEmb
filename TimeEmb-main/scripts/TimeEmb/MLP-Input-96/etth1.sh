model_name=TimeEmb

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
random_seed=2024
seq_len=96
for pred_len in 96 192
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
      --enc_in 7 \
      --use_hour_index 1 \
      --hour_length 24 \
      --train_epochs 30 \
      --patience 5 \
      --rec_lambda 0.25\
      --auxi_lambda 0.75\
      --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed $random_seed
done


for pred_len in 336
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
      --enc_in 7 \
      --use_hour_index 1 \
      --hour_length 24 \
      --train_epochs 30 \
      --patience 5 \
      --rec_lambda 0.\
      --auxi_lambda 1\
      --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed $random_seed
done


for pred_len in 720
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
      --enc_in 7 \
      --use_hour_index 1 \
      --hour_length 24 \
      --train_epochs 30 \
      --patience 5 \
      --rec_lambda 1\
      --auxi_lambda 0.\
      --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed $random_seed
done