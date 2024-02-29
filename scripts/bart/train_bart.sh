# CUDA_VISIBLE_DEVICES=5 python ./src/model/bart/train.py \
#     --data_path ./data/slotsum/clipped \
#     --output_path ./models/raw_bart/bart-large \
#     --base_model ./models/pretrained/bart-large \
#     --data_format json \
#     --seq_max_length 512 \
#     --cuda \
#     --additional_data none \
#     --learning_rate 1e-5 \
#     --epoch 4 \
#     --train_batch_size 8 \
#     --eval_batch_size 8 \
#     --print_interval 100 \
#     --seed 42

CUDA_VISIBLE_DEVICES=5 python ./src/model/bart/train.py \
    --data_path ./data/slotsum/clipped \
    --output_path ./models/bart_key/bart-large \
    --base_model ./models/pretrained/bart-large \
    --data_format json \
    --seq_max_length 512 \
    --cuda \
    --additional_data key \
    --learning_rate 1e-5 \
    --epoch 4 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --print_interval 100 \
    --seed 42

# CUDA_VISIBLE_DEVICES=5 python ./src/model/bart/train.py \
#     --data_path ./data/slotsum/clipped \
#     --output_path ./models/bart_kv/bart-large \
#     --base_model ./models/pretrained/bart-large \
#     --data_format json \
#     --seq_max_length 512 \
#     --cuda \
#     --additional_data kv \
#     --learning_rate 1e-5 \
#     --epoch 4 \
#     --train_batch_size 4 \
#     --eval_batch_size 8 \
#     --print_interval 100 \
#     --seed 42