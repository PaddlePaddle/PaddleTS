if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/NLinear" ]; then
    mkdir ./logs/NLinear
fi
seq_len=36
model_name=NLinear
data_name=ILI
config=./configs/longterm_forecast/NLinear_ILI.yaml

for pred_len in 24 36 48 60
do
    echo "training..."
    python3 -u train.py \
      --config $config \
      --seq_len $seq_len \
      --predict_len $pred_len \
      --save_dir ./logs/NLinear/ \
      --iters 1 >logs/NLinear/$model_name'_'$data_name'_'$seq_len'_'$pred_len.log 2>&1 &
done