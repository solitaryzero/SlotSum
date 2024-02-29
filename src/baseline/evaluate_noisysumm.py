from datasets import load_dataset, load_metric
import os
import json
import nltk


def main():
    raw_prediction_path = './data/slotsum/predictions/noisysumm/raw.txt'
    test_split_path = './data/slotsum/clipped/test.jsonl'
    merged_path = './data/slotsum/predictions/noisysumm/prediction.jsonl'

    answers = []
    with open(test_split_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            js = json.loads(line)
            answers.append(js['target_text'])

    claims = []
    with open(raw_prediction_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            claims.append(line.strip())

    assert len(answers) == len(claims)
    with open(merged_path, 'w', encoding='utf-8') as fout:
        for i, ans in enumerate(answers):
            js = {
                'id': i,
                'text': ans,
                'claim': claims[i]
            }
            fout.write(json.dumps(js))
            fout.write('\n')

    metric = load_metric('rouge')
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in claims]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in answers]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    res = {k: round(v, 4) for k, v in result.items()}
    print(res)


if __name__ == '__main__':
    main()