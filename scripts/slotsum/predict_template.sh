# CUDA_VISIBLE_DEVICES=5 python ./src/model/slotsum/predict_template.py \
#     --data_path ./data/slotsum/sloted \
#     --model_path ./models/slotsum/template/bart-large \
#     --output_path ./data/slotsum/predictions/template_prediction_nocnn \
#     --base_model ./models/pretrained/bart-large \
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

# CUDA_VISIBLE_DEVICES=5 python ./src/model/slotsum/predict_template.py \
#     --data_path ./data/slotsum/sloted \
#     --model_path ./models/slotsum/template/bart-large \
#     --output_path ./data/slotsum/predictions/template_prediction_new \
#     --base_model ./models/pretrained/bart-large \
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

# CUDA_VISIBLE_DEVICES=1 python ./src/model/slotsum/predict_template.py \
#     --data_path ./data/slotsum/sloted \
#     --model_path ./models/slotsum/template/bart-large_key \
#     --output_path ./data/slotsum/predictions/template_prediction_key \
#     --base_model ./models/pretrained/bart-large \
#     --data_format json \
#     --seq_max_length 512 \
#     --res_max_length 64 \
#     --cuda \
#     --use_key \
#     --learning_rate 1e-5 \
#     --epoch 4 \
#     --train_batch_size 8 \
#     --eval_batch_size 8 \
#     --print_interval 100 \
#     --seed 42

CUDA_VISIBLE_DEVICES=1 python ./src/model/slotsum/predict_template.py \
    --data_path ./data/slotsum/sloted \
    --model_path ./models/slotsum/template/bart-large_kv \
    --output_path ./data/slotsum/predictions/template_prediction_kv \
    --base_model ./models/pretrained/bart-large \
    --data_format json \
    --seq_max_length 512 \
    --res_max_length 64 \
    --cuda \
    --use_key \
    --learning_rate 1e-5 \
    --epoch 4 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --print_interval 100 \
    --seed 42