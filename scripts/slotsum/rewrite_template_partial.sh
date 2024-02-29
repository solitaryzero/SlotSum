CUDA_VISIBLE_DEVICES=1 python ./src/model/slotsum/fill_template.py \
    --data_path ./data/slotsum/sloted \
    --template_path ./data/slotsum/predictions/template_prediction_new \
    --filler_path ./models/slotsum/filler/bart-large \
    --data_format json \
    --data_split test \
    --threshold 80 \
    --missing_strategy predict \
    --fact_ratio 0.5 \
    --seq_max_length 512 \
    --cuda \
    --seed 42

CUDA_VISIBLE_DEVICES=2 python ./src/model/slotsum/rewrite_template.py \
    --data_path ./data/slotsum/sloted \
    --template_path ./data/slotsum/predictions/template_prediction_new \
    --filler_path ./models/slotsum/filler/bart-large \
    --rewriter_path ./models/slotsum/rewriter_source/bart-large \
    --mode source \
    --facts partial \
    --fact_ratio 0.5 \
    --data_format json \
    --data_split test \
    --threshold 80 \
    --seq_max_length 512 \
    --res_max_length 64 \
    --cuda \
    --seed 42
