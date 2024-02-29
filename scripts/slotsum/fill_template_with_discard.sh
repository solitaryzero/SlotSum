# python ./src/model/slotsum/fill_template.py \
#     --data_path ./data/slotsum/sloted \
#     --template_path ./data/slotsum/predictions/template_prediction \
#     --data_format json \
#     --data_split test \
#     --threshold 80 \
#     --missing_strategy discard \
#     --seed 42

# python ./src/model/slotsum/fill_template.py \
#     --data_path ./data/slotsum/sloted \
#     --template_path ./data/slotsum/predictions/template_prediction_nocnn \
#     --data_format json \
#     --data_split test \
#     --threshold 80 \
#     --missing_strategy discard \
#     --seed 42

python ./src/model/slotsum/fill_template.py \
    --data_path ./data/slotsum/sloted \
    --template_path ./data/slotsum/predictions/template_prediction_kv \
    --data_format json \
    --data_split test \
    --threshold 80 \
    --missing_strategy discard \
    --seed 42