CUDA_VISIBLE_DEVICES=0 python ./src/model/slotsum/rewrite_template.py \
    --data_path ./data/slotsum/sloted \
    --template_path ./data/slotsum/predictions/template_prediction_kv \
    --filler_path ./models/slotsum/filler/bart-large \
    --rewriter_path ./models/slotsum/rewriter/bart-large \
    --mode template \
    --facts all \
    --template predict \
    --data_format json \
    --data_split test \
    --threshold 80 \
    --seq_max_length 512 \
    --res_max_length 64 \
    --cuda \
    --seed 42

# CUDA_VISIBLE_DEVICES=6 python ./src/model/slotsum/rewrite_template.py \
#     --data_path ./data/slotsum/sloted \
#     --template_path ./data/slotsum/predictions/template_prediction_nocnn \
#     --filler_path ./models/slotsum/filler/bart-large \
#     --rewriter_path ./models/slotsum/rewriter_source/bart-large \
#     --mode source \
#     --facts all \
#     --data_format json \
#     --data_split test \
#     --threshold 80 \
#     --seq_max_length 512 \
#     --res_max_length 64 \
#     --cuda \
#     --seed 42
