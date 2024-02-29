# python ./src/bertscore/score_selected.py \
#     --prediction_path ./data/slotsum/predictions/template_prediction_kv/filled_predictions_discard.jsonl \
#     --out_path ./data/stats/any.json \
#     --seed 42

# python ./src/bertscore/score_selected.py \
#     --prediction_path ./data/slotsum/predictions/template_prediction_kv/filled_predictions_predict.jsonl \
#     --out_path ./data/stats/any.json \
#     --seed 42

python ./src/bertscore/score_selected.py \
    --prediction_path ./data/slotsum/predictions/template_prediction_kv/filled_predictions_allpredict.jsonl \
    --out_path ./data/stats/any.json \
    --seed 42

# python ./src/bertscore/score_selected.py \
#     --prediction_path ./data/slotsum/predictions/bart_kv/prediction.jsonl \
#     --out_path ./data/stats/any.json \
#     --seed 42

# python ./src/bertscore/score_selected.py \
#     --prediction_path ./data/slotsum/predictions/t5-base_kv/prediction.jsonl \
#     --out_path ./data/stats/any.json \
#     --seed 42

# python ./src/bertscore/score_selected.py \
#     --prediction_path ./data/slotsum/predictions/pegasus_kv/prediction.jsonl \
#     --out_path ./data/stats/any.json \
#     --seed 42