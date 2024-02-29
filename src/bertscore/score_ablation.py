import os
import json
import argparse
from tqdm import tqdm
from bert_score import score


def main(params):
    # TODO
    prediction_paths = {
        'Fill (P-Template)': './data/slotsum/predictions/template_prediction_nocnn/filled_predictions_discard.jsonl',
        'Fill (G-Template)': './data/slotsum/predictions/template_prediction_nocnn/filled_predictions_discard.jsonl',
        'Rewrite (P-Template)': './data/slotsum/predictions/template_prediction_nocnn/filled_predictions_discard.jsonl',
        'Rewrite (G-Template)': './data/slotsum/predictions/template_prediction_nocnn/filled_predictions_discard.jsonl',
        'Rewrite (Source)': './data/slotsum/predictions/template_prediction_nocnn/filled_predictions_discard.jsonl',
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