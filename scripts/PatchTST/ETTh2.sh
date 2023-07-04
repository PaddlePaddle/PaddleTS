if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/PatchTST" ]; then
    mkdir ./logs/PatchTST
fi
seq_len=96
model_name=PatchTST
data_name=ETTH2
config=./configs/longterm_forecast/PatchTST_Etth2.yaml

for pred_len in 96 192 336 720
do
    echo "training..."
    python3 -u train.py \
      --config $config \
      --seq_len $seq_len \
      --predict_len $pred_len \
      --save_dir ./logs/PatchTST/ \
      --iters 1 >logs/PatchTST/$model_name'_'$data_name'_'$seq_len'_'$pred_len.log 2>&1 &
done