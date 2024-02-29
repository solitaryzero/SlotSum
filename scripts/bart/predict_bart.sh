# CUDA_VISIBLE_DEVICES=5 python ./src/model/bart/predict.py \
#     --data_path ./data/slotsum/clipped \
#     --model_path ./models/raw_bart/bart-large \
#     --output_path ./data/slotsum/predictions/bart_raw \
#     --base_model ./models/pretrained/bart-large \
#     --data_format json \
#     --seq_max_length 512 \
#     --res_max_length 64 \
#     --additional_data none \
#     --cuda \
#     --learning_rate 1e-5 \
#     --epoch 4 \
#     --train_batch_size 8 \
#     --eval_batch_size 8 \
#     --print_interval 100 \
#     --seed 42

CUDA_VISIBLE_DEVICES=0 python ./src/model/bart/predict.py \
    --data_path ./data/slotsum/clipped \
    --model_path ./models/bart_key/bart-large \
    --output_path ./data/slotsum/predictions/bart_key \
    --base_model ./models/pretrained/bart-large \
    --data_format json \
    --seq_max_length 512 \
    --res_max_length 64 \
    --additional_data key \
    --cuda \
    --learning_rate 1e-5 \
    --epoch 4 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --print_interval 100 \
    --seed 42

# CUDA_VISIBLE_DEVICES=5 python ./src/model/bart/predict.py \
#     --data_path ./data/slotsum/clipped \
#     --model_path ./models/bart_kv/bart-large \
#     --output_path ./data/slotsum/predictions/bart_kv \
#     --base_model ./models/pretrained/bart-large \
#     --data_format json \
#     --seq_max_length 512 \
#     --res_max_length 64 \
#     --additional_data kv \
#     --cuda \
#     --learning_rate 1e-5 \
#     --epoch 4 \
#     --train_batch_size 8 \
#     --eval_batch_size 8 \
#     --print_interval 100 \
#     --seed 42