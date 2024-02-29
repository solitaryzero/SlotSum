# CUDA_VISIBLE_DEVICES=4 python ./src/model/slotsum/train_template.py \
#     --data_path ./data/slotsum/sloted \
#     --output_path ./models/slotsum/template/bart-large \
#     --base_model ./models/pretrained/bart-large \
#     --data_format json \
#     --seq_max_length 512 \
#     --cuda \
#     --learning_rate 1e-5 \
#     --epoch 4 \
#     --train_batch_size 8 \
#     --eval_batch_size 8 \
#     --print_interval 100 \
#     --seed 42

# CUDA_VISIBLE_DEVICES=1 python ./src/model/slotsum/train_template.py \
#     --data_path ./data/slotsum/sloted \
#     --output_path ./models/slotsum/template/bart-large_key \
#     --base_model ./models/pretrained/bart-large \
#     --data_format json \
#     --seq_max_length 512 \
#     --cuda \
#     --use_key \
#     --learning_rate 1e-5 \
#     --epoch 4 \
#     --train_batch_size 4 \
#     --eval_batch_size 8 \
#     --print_interval 100 \
#     --seed 42

CUDA_VISIBLE_DEVICES=1 python ./src/model/slotsum/train_template.py \
    --data_path ./data/slotsum/sloted \
    --output_path ./models/slotsum/template/bart-large_kv \
    --base_model ./models/pretrained/bart-large \
    --data_format json \
    --seq_max_length 512 \
    --cuda \
    --use_key \
    --learning_rate 1e-5 \
    --epoch 4 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --print_interval 100 \
    --seed 42