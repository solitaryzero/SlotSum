import os
import subprocess
import argparse

if __name__ == '__main__':
    prediction_paths = {
        # raw models
        # 'BART': './data/slotsum/predictions/bart_raw/prediction.jsonl',
        # 'T5': './data/slotsum/predictions/t5-base/prediction.jsonl',
        # 'Pegasus': './data/slotsum/predictions/pegasus/prediction.jsonl',
        # 'NoisySumm': './data/slotsum/predictions/noisysumm/prediction.jsonl',
        # 'SlotSum (Discard)': './data/slotsum/predictions/template_prediction_nocnn/filled_predictions_discard.jsonl',
        # 'SlotSum (Predict)': './data/slotsum/predictions/template_prediction_nocnn/filled_predictions_predict.jsonl',
        # 'SlotSum (All Predict)': './data/slotsum/predictions/template_prediction_nocnn/filled_predictions_allpredict.jsonl',
        
        # models with K
        # 'BART+K': './data/slotsum/predictions/bart_key/prediction.jsonl',
        # 'T5+K': './data/slotsum/predictions/t5-base_key/prediction.jsonl',
        # 'Pegasus+K': './data/slotsum/predictions/pegasus_key/prediction.jsonl',
        # 'SlotSum (Discard)+K': './data/slotsum/predictions/template_prediction_key/filled_predictions_discard.jsonl',
        # 'SlotSum (Predict)+K': './data/slotsum/predictions/template_prediction_key/filled_predictions_predict.jsonl',
        # 'SlotSum (All Predict)+K': './data/slotsum/predictions/template_prediction_key/filled_predictions_allpredict.jsonl',

        # models with K+V
        # 'BART+KV': './data/slotsum/predictions/bart_kv/prediction.jsonl',
        # 'T5+KV': './data/slotsum/predictions/t5-base_kv/prediction.jsonl',
        # 'Pegasus+KV': './data/slotsum/predictions/pegasus_kv/prediction.jsonl',
        # 'SlotSum (Discard)+KV': './data/slotsum/predictions/template_prediction_kv/filled_predictions_discard.jsonl',
        # 'SlotSum (Predict)+KV': './data/slotsum/predictions/template_prediction_kv/filled_predictions_predict.jsonl',
        # 'SlotSum (All Predict)+KV': './data/slotsum/predictions/template_prediction_kv/filled_predictions_allpredict.jsonl',
    
        # oracle model
        'SlotSum (Oracle)': './data/slotsum/predictions/template_prediction_golden/filled_predictions_predict.jsonl',
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='./data/stats/factcc')
    args = parser.parse_args()

    if not(os.path.exists(args.out_path)):
        os.makedirs(args.out_path)

    for model in prediction_paths:
        path = prediction_paths[model]
        command = f'rm ../factCC/data/cached_dev_bert-base-uncased_512_factcc_annotated; cp {path} ../factCC/data/data-dev.jsonl; sh ../factCC/modeling/scripts/factcc-eval.sh'
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT).decode('utf-8')
        output_path = os.path.join(args.out_path, f'{model}.txt')
        with open(output_path, 'w', encoding='utf-8') as fout:
            fout.write(output)