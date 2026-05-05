#!/bin/bash

# 数据集列表
# datasets=("IAC1" "IAC2" "Twitter")
datasets=("IAC2")

# 通用参数
voc_size=30000
batch_size=128
per_checkpoint=64
n_class=2
embed_dropout_rate=0.5
max_length_sen=100
n_layers=3
model_name=dualbilstm

# 日志 & 模型目录
log_dir="logs"
model_dir="models"

mkdir -p $log_dir
mkdir -p $model_dir

# 时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")

# 遍历每个数据集
for dataset in "${datasets[@]}"
do
    echo "🔁 正在运行数据集: $dataset"

    # 每个数据集独立日志文件
    log_file="${log_dir}/${dataset}_${model_name}_${timestamp}.log"
    model_path="${model_dir}/${dataset}_${model_name}"

    mkdir -p $model_path

    CUDA_VISIBLE_DEVICES=0 python main.py \
        --voc_size $voc_size \
        --data_dir ./${dataset}/spacy/ \
        --name_dataset $dataset \
        --breakpoint -1 \
        --batch_size $batch_size \
        --per_checkpoint $per_checkpoint \
        --n_class $n_class \
        --embed_dropout_rate $embed_dropout_rate \
        --name_model $model_name \
        --max_length_sen $max_length_sen \
        --n_layers $n_layers \
        --model_dir $model_path/ \
        --save_model 0 \
        > $log_file 2>&1

    echo "✅ 完成数据集: $dataset，日志保存在 $log_file"
done
