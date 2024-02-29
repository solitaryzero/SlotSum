import os
import json
import argparse
from tqdm import tqdm
from bert_score import score


def main(params):
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
    
    results = {}

    for model_name in prediction_paths:
        print('Predicting BertScore of %s' %model_name)

        full_path = prediction_paths[model_name]
        all_predictions = []
        all_refs = []
        with open(full_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                js = json.loads(line)
                ref = js['text']
                pred = js['claim']
                all_refs.append(ref)
                all_predictions.append(pred)

        P, R, F1 = score(all_predictions, all_refs, lang='en', verbose=True)
        res = {
            'P': P.mean().item(),
            'R': R.mean().item(),
            'F': F1.mean().item(),
        }

        results[model_name] = res

    if not(os.path.exists(params['out_path'])):
        os.makedirs(params['out_path'])

    with open(os.path.join(params['out_path'], 'bertscore.json'), 'w', encoding='utf-8') as fout:
        json.dump(results, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='./data/stats')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    main(params)