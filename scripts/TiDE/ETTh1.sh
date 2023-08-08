if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/TiDE" ]; then
    mkdir ./logs/TiDE
fi
seq_len=720
model_name=TiDE
data_name=ETTH1
config=./configs/longterm_forecast/TiDE_Etth1.yaml

for pred_len in 96 192
do
    echo "training..."
    python3 -u train.py \
      --config $config \
      --seq_len $seq_len \
      --predict_len $pred_len \
      --batch_size 4 \
      --time_feat \
      --save_dir ./logs/TiDE/ \
      --iters 1 >logs/TiDE/$model_name'_'$data_name'_'$seq_len'_'$pred_len.log 2>&1 &
done

seq_len=336
for pred_len in 336 720
do
    echo "training..."
    python3 -u train.py \
      --config $config \
      --seq_len $seq_len \
      --predict_len $pred_len \
      --batch_size 1 \
      --time_feat \
      --save_dir ./logs/TiDE/ \
      --opts model.model_cfg.drop_prob=0.3 \
            model.model_cfg.hidden_size=256  \
            model.model_cfg.decoder_output_dim=8 \
            model.model_cfg.temporal_decoder_hidden=128 \
            model.model_cfg.optimizer_params.learning_rate=0.00005 \
            model.model_cfg.patience=20 \
      --iters 1 >logs/TiDE/$model_name'_'$data_name'_'$seq_len'_'$pred_len'_1'.log 2>&1 &
done