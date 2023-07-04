if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/RLinear" ]; then
    mkdir ./logs/RLinear
fi
seq_len=96
model_name=RLinear
data_name=ETTH1
config=./configs/longterm_forecast/RLinear_Etth1.yaml

for pred_len in 96 192 336 720
do
    echo "training..."
    python3 -u train.py \
      --config $config \
      --seq_len $seq_len \
      --predict_len $pred_len \
      --save_dir ./logs/RLinear/ \
      --iters 1 >logs/RLinear/$model_name'_'$data_name'_'$seq_len'_'$pred_len.log 2>&1 &
done