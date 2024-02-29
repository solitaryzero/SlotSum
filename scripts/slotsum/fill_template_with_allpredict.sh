# CUDA_VISIBLE_DEVICES=4 python ./src/model/slotsum/fill_template.py \
#     --data_path ./data/slotsum/sloted \
#     --template_path ./data/slotsum/predictions/template_prediction \
#     --filler_path ./models/slotsum/filler/bart-large \
#     --data_format json \
#     --data_split test \
#     --threshold 80 \
#     --missing_strategy allpredict \
#     --seq_max_length 512 \
#     --cuda \
#     --seed 42

# CUDA_VISIBLE_DEVICES=4 python ./src/model/slotsum/fill_template.py \
#     --data_path ./data/slotsum/sloted \
#     --template_path ./data/slotsum/predictions/template_prediction_nocnn \
#     --filler_path ./models/slotsum/filler/bart-large \
#     --data_format json \
#     --data_split test \
#     --threshold 80 \
#     --missing_strategy allpredict \
#     --seq_max_length 512 \
#     --cuda \
#     --seed 42

# CUDA_VISIBLE_DEVICES=4 python ./src/model/slotsum/fill_template.py \
#     --data_path ./data/slotsum/sloted \
#     --template_path ./data/slotsum/predictions/template_prediction_key \
#     --filler_path ./models/slotsum/filler/bart-large \
#     --data_format json \
#     --data_split test \
#     --threshold 80 \
#     --missing_strategy allpredict \
#     --seq_max_length 512 \
#     --cuda \
#     --seed 42

CUDA_VISIBLE_DEVICES=1 python ./src/model/slotsum/fill_template.py \
    --data_path ./data/slotsum/sloted \
    --template_path ./data/slotsum/predictions/template_prediction_kv \
    --filler_path ./models/slotsum/filler/bart-large \
    --data_format json \
    --data_split test \
    --threshold 80 \
    --missing_strategy allpredict \
    --seq_max_length 512 \
    --cuda \
    --seed 42