if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Nonstationary" ]; then
    mkdir ./logs/Nonstationary
fi
seq_len=96
model_name=Nonstationary
data_name=weather
config=./configs/longterm_forecast/Nonstationary_weather.yaml

for pred_len in 96 192 336 720
do
    echo "training..."
    python3 -u train.py \
      --config $config \
      --seq_len $seq_len \
      --predict_len $pred_len \
      --time_feat \
      --save_dir ./logs/Nonstationary/ \
      --iters 1 >logs/Nonstationary/$model_name'_'$data_name'_'$seq_len'_'$pred_len.log 2>&1 &
done