import os
import json
import argparse
from tqdm import tqdm
from bert_score import score


def main(params):
    full_path = os.path.join(params['prediction_path'])
    
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
        'P': P.mean(),
        'R': R.mean(),
        'F': F1.mean(),
    }

    print('BertScore: ')
    print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--prediction_path', type=str, default='./data/slotsum/predictions/bart_raw/prediction.jsonl')
    parser.add_argument('--out_path', type=str, default='./data/stats/any.json')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    main(params)