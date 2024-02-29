# CUDA_VISIBLE_DEVICES=5 python ./src/model/t5/evaluate.py \
#     --data_path ./data/slotsum/clipped \
#     --model_path ./models/t5-base \
#     --base_model ./models/pretrained/t5-base \
#     --data_format json \
#     --seq_max_length 512 \
#     --res_max_length 64 \
#     --cuda \
#     --learning_rate 1e-5 \
#     --epoch 4 \
#     --train_batch_size 8 \
#     --eval_batch_size 8 \
#     --print_interval 100 \
#     --seed 42

CUDA_VISIBLE_DEVICES=0 python ./src/model/t5/evaluate.py \
    --data_path ./data/slotsum/clipped \
    --model_path ./models/t5-base_key \
    --base_model ./models/pretrained/t5-base \
    --data_format json \
    --seq_max_length 512 \
    --res_max_length 64 \
    --cuda \
    --additional_data key \
    --learning_rate 1e-5 \
    --epoch 4 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --print_interval 100 \
    --seed 42

# CUDA_VISIBLE_DEVICES=0 python ./src/model/t5/evaluate.py \
#     --data_path ./data/slotsum/clipped \
#     --model_path ./models/t5-base_kv \
#     --base_model ./models/pretrained/t5-base \
#     --data_format json \
#     --seq_max_length 512 \
#     --res_max_length 64 \
#     --cuda \
#     --additional_data kv \
#     --learning_rate 1e-5 \
#     --epoch 4 \
#     --train_batch_size 4 \
#     --eval_batch_size 8 \
#     --print_interval 100 \
#     --seed 42