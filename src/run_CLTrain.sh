#!/bin/bash

# NUM_GPU=1

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
while getopts  m:c:d:e:b:r:y:g:t:p: flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        m) model_name_or_path=${OPTARG};;
        c) c_model_name_or_path=${OPTARG};;
        d) train_file=${OPTARG};;
        e) num_train_epochs=${OPTARG};;
        b) per_device_train_batch_size=${OPTARG};;
        r) learning_rate=${OPTARG};;
        y) seed=${OPTARG};;
        g) NUM_GPU=${OPTARG};;
        t) eval_step=${OPTARG};;
        p) phi=${OPTARG};;
    esac
done

#if [ $sup == "sup" ]
#then
#    file_name="data_simcse/${train_file}_for_simcse.csv"
#    f_name="simcse_result/sup-${train_file}-${seed}-${per_device_train_batch_size}-${phi}"
#
#    # Randomly set a port number
#    # If you encounter "address already used" error, just run again or manually set an available port id.
#    PORT_ID=$(expr $RANDOM + 1000)
#
#    # Allow multiple threads
#    export OMP_NUM_THREADS=8
#    # export TRANSFORMERS_OFFLINE=1
#
#    python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
#    --train_file $file_name --output_dir $f_name \
#    --num_train_epochs $num_train_epochs --per_device_train_batch_size $per_device_train_batch_size \
#    --learning_rate $learning_rate --evaluation_strategy steps \
#    --metric_for_best_model stsb_spearman --load_best_model_at_end --eval_steps $eval_step \
#    --overwrite_output_dir --temp 0.05 --seed $seed --do_train --do_eval --phi $phi\
#    # --fp16 \
#    "$@"
#else

# 设置默认参数值
model_name_or_path="/root/LLM-R2/models/" 
# train_file="queries/queries_job_syn_train.csv"                          # 训练文件名，例如wiki1m
train_file="queries/queries_tpch_train.csv"                          # 训练文件名，例如wiki1m
num_train_epochs=3                          # 训练轮数，例如3轮
per_device_train_batch_size=8               # 每个设备的批次大小，例如64
learning_rate="1e-5"                         # 学习率，例如3e-5
seed=42                                      # 随机种子，例如42
NUM_GPU=1                                    # GPU数量，例如使用1张GPU
eval_step=100                                # 评估步数，例如每100步评估
phi=0.05                                     # 温度值，例如0.05
##    file_name="data_simcse/${train_file}_for_simcse.txt"
file_name="../data/data_llmr2/${train_file}"
f_name="simcse_models/sup-${train_file}-${seed}-${per_device_train_batch_size}-${phi}"
# python train.py --train_file $file_name --output_dir $f_name \
# --num_train_epochs $num_train_epochs --per_device_train_batch_size $per_device_train_batch_size \
# --learning_rate $learning_rate --evaluation_strategy no \
# --model_name_or_path $model_name_or_path \
#--metric_for_best_model stsb_spearman \
# --load_best_model_at_end 
# --eval_steps $eval_step \
# --overwrite_output_dir 
# --temp 0.05 --seed $seed \
# --do_train \
# --do_eval=True \
# --fp16 \
# "$@"

#fi
python train.py \
--train_file $file_name --output_dir $output_dir  $f_name \
--num_train_epochs $num_train_epochs --per_device_train_batch_size $per_device_train_batch_size \
--learning_rate $learning_rate --evaluation_strategy steps \
--eval_steps $eval_step --do_train --do_eval --seed $seed --phi $phi \
--temp $phi "$@"
# python evaluation.py --pooler cls_before_pooler --task_set sts --mode test --model_name_or_path result/unsup-bert-base-uncased-3407
# python evaluation.py --pooler cls --task_set sts --mode test --model_name_or_path result/sup-nli-bert-base-uncased-3407

# bash run.sh -s unsup -m bert-base-uncased -d wiki1m -e 1 -b 64 -r 3e-5 -l 32 -y 42 -g 1 -t 25
# bash run.sh -s sup -m bert-base-uncased -d nli -e 3 -b 128 -r 5e-5 -l 32 -y 42 -g 2 -t 25

# bash run_CLTrain.sh -s sup -d testfile -e 3 -r 5e-5 -l 32 -y 42 -g 1 -t 125 -p 0 -b 8