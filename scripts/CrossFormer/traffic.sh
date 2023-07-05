if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Crossformer" ]; then
    mkdir ./logs/Crossformer
fi
seq_len=96
model_name=Crossformer
data_name=traffic
config=./configs/longterm_forecast/Crossformer_traffic.yaml

for pred_len in 96 192 336 720
do
    echo "training..."
    python3 -u train.py \
      --config $config \
      --seq_len $seq_len \
      --predict_len $pred_len \
      --save_dir ./logs/Crossformer/ \
      --iters 1 >logs/Crossformer/$model_name'_'$data_name'_'$seq_len'_'$pred_len.log 2>&1 &
done