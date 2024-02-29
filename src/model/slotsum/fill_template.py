import os
import argparse
import numpy as np
import re
import nltk
import json
from tqdm import tqdm
from thefuzz import fuzz
from thefuzz import process
import random

from datasets import load_dataset, load_metric

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,  
)

def load_slotsum_dataset(
    dataset_path, 
    data_format='json', 
    splits=['train', 'validation', 'test']
):
    data_files = {}
    if (data_format == 'json'):
        for spl in splits:
            data_files[spl] = os.path.join(dataset_path, '%s.jsonl' %spl )
        dataset = load_dataset(data_format, data_files=data_files)
    else:
        dataset = load_dataset(dataset_path)

    return dataset


def load_predicted_templates(
    template_path,
    data_format='json', 
):
    data_files = {
        'prediction': os.path.join(template_path, 'prediction.jsonl')
    }
    template_set = load_dataset(data_format, data_files=data_files)
    return template_set


def load_golden_templates(
    golden_dataset,
    data_split,
):
    templates = []
    for data in golden_dataset[data_split]:
        t = {'template_prediction': data['sloted_text']}
        templates.append(t)
    
    return templates


def predict_query(
    query,
    source_text,
    filler_model,
    filler_tokenizer,
    cuda=True
):
    assert (source_text is not None) and (filler_model is not None) and (filler_tokenizer is not None)
    model_inputs = filler_tokenizer([[query, source_text]], truncation=True, return_tensors="pt")['input_ids']
    if (cuda):
        model_inputs = model_inputs.to('cuda')
    outputs = filler_model.generate(model_inputs, num_beams=5)[0] # beam search
    # outputs = filler_model.generate(model_inputs, num_beams=1, do_sample=False)[0] # greedy search
    return filler_tokenizer.decode(outputs, skip_special_tokens=True)


def fill_template(
    params,
    template, 
    table,
    threshold=80,
    strategy='discard',
    source_text=None,
    title='',
    filler_model=None,
    filler_tokenizer=None,
):
    pattern = re.compile(r'\#\#.+?\#\#')
    res = pattern.finditer(template)
    slots = []
    for x in res:
        start, end = x.start(), x.end()
        slots.append((start, end, template[start+2: end-2]))
    slots.sort(reverse=True)

    candidates = list(table.keys())
    if (params['fact_ratio'] < 1.0):
        candidates = random.sample(candidates, int(len(candidates)*params['fact_ratio']))

    summary = template
    for slot in slots:
        start, end, query = slot
        if (len(candidates) == 0): # no data provided
            if (strategy == 'discard'):
                summary = summary[:start] + summary[end:]
            elif (strategy == 'keep'):
                pass
            elif (strategy == 'predict') or (strategy == 'allpredict'):
                prediction = predict_query(
                    title + ' ' + query, source_text, filler_model, filler_tokenizer
                )
                summary = summary[:start] + prediction + summary[end:]
        else:
            if (strategy == 'allpredict'):
                prediction = predict_query(
                    title + ' ' + query, source_text, filler_model, filler_tokenizer
                )
                summary = summary[:start] + prediction + summary[end:]
            else:
                choice, score = process.extractOne(query, candidates, scorer=fuzz.token_sort_ratio)
                # print(query)
                # print(choice, score)
                if (score >= threshold):
                    summary = summary[:start] + table[choice] + summary[end:]
                elif (strategy == 'discard'):
                    summary = summary[:start] + summary[end:]
                elif (strategy == 'keep'):
                    pass
                elif (strategy == 'predict'):
                    prediction = predict_query(
                        query, source_text, filler_model, filler_tokenizer
                    )
                    summary = summary[:start] + prediction + summary[end:]

    return summary


def main(params):
    random.seed(params['seed'])

    slotsum_dataset = load_slotsum_dataset(params['data_path'], params['data_format'])
    if (params['template'] == 'predict'):
        template_set = load_predicted_templates(params['template_path'], params['data_format'])['prediction']
    else:
        template_set = load_golden_templates(slotsum_dataset, params['data_split'])
    metric = load_metric('rouge')
    data_split = params['data_split']

    if (params['missing_strategy'] == 'predict') or (params['missing_strategy'] == 'allpredict'):
        assert (params['filler_path'] is not None)
        tokenizer = AutoTokenizer.from_pretrained(params['filler_path'], model_max_length=params['seq_max_length'])
        model = AutoModelForSeq2SeqLM.from_pretrained(params['filler_path'])
        if (params['cuda']):
            model = model.to('cuda')
    else:
        tokenizer, model = None, None

    def compute_metrics(preds, labels):
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        return {k: round(v, 4) for k, v in result.items()}

    bad_headers = [
        'article_title',
        'birth_name',
        'image', 
        'imagesize',
        'bgcolour', 
        'caption',
    ]

    labels = [example['target_text'] for example in slotsum_dataset[data_split]]
    preds = []
    for i, example in tqdm(enumerate(template_set)):
        kv_info = slotsum_dataset[data_split][i]['kv_info']
        table = {}
        for header, content in zip(kv_info['column_header'], kv_info['content']):
            if (header not in bad_headers):
                table[header] = content
        source_text = slotsum_dataset[data_split][i]['source_text']
        title = slotsum_dataset[data_split][i]['title']

        template = example['template_prediction']
        filled_template = fill_template(
            params=params,
            template=template, 
            table=table, 
            source_text=source_text,
            title=title,
            threshold=params['threshold'],
            strategy=params['missing_strategy'],
            filler_tokenizer=tokenizer,
            filler_model=model,
        )
        # print(template)
        # print(filled_template)
        # input()
        preds.append(filled_template)

    if (params['template'] == 'golden'):
        if not(os.path.exists(params['template_path'])):
            os.makedirs(params['template_path'])

    # save eval results
    results = compute_metrics(preds, labels)
    print(results)
    out_path = os.path.join(params['template_path'], 'eval_results_%s.json' %params['missing_strategy'])
    # out_path = os.path.join(params['template_path'], 'eval_results_%s_%f.json' %(params['missing_strategy'], params['fact_ratio']))
    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, indent=4)

    # save predictions for factCC
    out_path = os.path.join(params['template_path'], 'filled_predictions_%s.jsonl' %params['missing_strategy'])
    # out_path = os.path.join(params['template_path'], 'filled_predictions_%s_%f.jsonl' %(params['missing_strategy'], params['fact_ratio']))
    with open(out_path, 'w', encoding='utf-8') as fout:
        for i, (pred, label) in enumerate(zip(preds, labels)):
            out_js = {
                'id': i,
                'text': label,
                'claim': pred,
            }
            json.dump(out_js, fout, ensure_ascii=False)
            fout.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--data_path', type=str, default='./data/slotsum/sloted')
    parser.add_argument('--template_path', type=str, default='./data/slotsum/template_prediction')
    parser.add_argument('--filler_path', type=str, default=None)
    parser.add_argument('--data_format', type=str, default="json")
    parser.add_argument('--data_split', type=str, default="test", choices=['test', 'validation'])
    parser.add_argument('--threshold', type=int, default=80)
    
    parser.add_argument('--missing_strategy', type=str, choices=['discard', 'keep', 'predict', 'allpredict'], default='discard')
    parser.add_argument('--template', type=str, choices=['predict', 'golden'], default='predict')
    parser.add_argument('--fact_ratio', type=float, default=1.0)
    parser.add_argument('--seq_max_length', type=int, default=512)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    main(params)