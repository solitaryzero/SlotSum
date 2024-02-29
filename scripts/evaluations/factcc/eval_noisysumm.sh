rm ../factCC/data/cached_dev_bert-base-uncased_512_factcc_annotated
cp ./data/slotsum/predictions/noisysumm/prediction.jsonl ../factCC/data/data-dev.jsonl
sh ../factCC/modeling/scripts/factcc-eval.sh