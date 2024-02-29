# rm ../factCC/data/cached_dev_bert-base-uncased_512_factcc_annotated
# cp ./data/slotsum/predictions/template_prediction_nocnn/filled_predictions_discard.jsonl ../factCC/data/data-dev.jsonl
# sh ../factCC/modeling/scripts/factcc-eval.sh

rm ../factCC/data/cached_dev_bert-base-uncased_512_factcc_annotated
cp ./data/slotsum/predictions/template_prediction_nocnn/filled_predictions_predict.jsonl ../factCC/data/data-dev.jsonl
sh ../factCC/modeling/scripts/factcc-eval.sh

# rm ../factCC/data/cached_dev_bert-base-uncased_512_factcc_annotated
# cp ./data/slotsum/predictions/template_prediction_nocnn/filled_predictions_allpredict.jsonl ../factCC/data/data-dev.jsonl
# sh ../factCC/modeling/scripts/factcc-eval.sh