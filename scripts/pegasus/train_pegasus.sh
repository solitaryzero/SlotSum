# CUDA_VISIBLE_DEVICES=5 python ./src/model/pegasus/train.py \
#     --data_path ./data/slotsum/clipped \
#     --output_path ./models/pegasus \
#     --base_model ./models/pretrained/pegasus-xsum \
#     --data_format json \
#     --seq_max_length 512 \
#     --cuda \
#     --additional_data none \
#     --learning_rate 1e-5 \
#     --epoch 4 \
#     --train_batch_size 8 \
#     --eval_batch_size 8 \
#     --print_interval 100 \
#     --seed 123

CUDA_VISIBLE_DEVICES=1 python ./src/model/pegasus/train.py \
    --data_path ./data/slotsum/clipped \
    --output_path ./models/pegasus_key \
    --base_model ./models/pretrained/pegasus-xsum \
    --data_format json \
    --seq_max_length 512 \
    --cuda \
    --additional_data key \
    --learning_rate 1e-5 \
    --epoch 4 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --print_interval 100 \
    --seed 123

# CUDA_VISIBLE_DEVICES=1 python ./src/model/pegasus/train.py \
#     --data_path ./data/slotsum/clipped \
#     --output_path ./models/pegasus_kv \
#     --base_model ./models/pretrained/pegasus-xsum \
#     --data_format json \
#     --seq_max_length 512 \
#     --cuda \
#     --additional_data kv \
#     --learning_rate 1e-5 \
#     --epoch 4 \
#     --train_batch_size 4 \
#     --eval_batch_size 8 \
#     --print_interval 100 \
#     --seed 123