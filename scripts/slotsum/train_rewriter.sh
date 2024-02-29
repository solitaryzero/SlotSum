# CUDA_VISIBLE_DEVICES=0 python ./src/model/slotsum/train_rewriter.py \
#     --data_path ./data/slotsum/rewriter \
#     --output_path ./models/slotsum/rewriter/bart-large \
#     --base_model ./models/pretrained/bart-large \
#     --data_format json \
#     --seq_max_length 512 \
#     --mode template \
#     --cuda \
#     --learning_rate 1e-5 \
#     --epoch 4 \
#     --train_batch_size 4 \
#     --eval_batch_size 8 \
#     --print_interval 100 \
#     --seed 42


CUDA_VISIBLE_DEVICES=1 python ./src/model/slotsum/train_rewriter.py \
    --data_path ./data/slotsum/rewriter \
    --output_path ./models/slotsum/rewriter_source/bart-large \
    --base_model ./models/pretrained/bart-large \
    --data_format json \
    --seq_max_length 512 \
    --mode source \
    --cuda \
    --learning_rate 1e-5 \
    --epoch 4 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --print_interval 100 \
    --seed 42