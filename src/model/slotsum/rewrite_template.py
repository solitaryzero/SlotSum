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
    cuda=True,
):
    assert (source_text is not None) and (filler_model is not None) and (filler_tokenizer is not None)
    model_inputs = filler_tokenizer([[query, source_text]], truncation=True, return_tensors="pt")['input_ids']
    if (cuda):
        model_inputs = model_inputs.to('cuda')
    outputs = filler_model.generate(model_inputs, num_beams=5)[0] # beam search
    # outputs = filler_model.generate(model_inputs, num_beams=1, do_sample=False)[0] # greedy search
    return filler_tokenizer.decode(outputs, skip_special_tokens=True)


def rewrite(
    template,
    source,
    facts,
    res_max_length=64,
    rewriter_model=None,
    rewriter_tokenizer=None,
    cuda=True,
):
    t = []
    for key, value in facts:
        t.append('%s | %s' %(key, value))
    kv_str = ' # '.join(t)

    if (template is not None):
        model_inputs = rewriter_tokenizer([[kv_str, template]], truncation=True, return_tensors="pt")['input_ids']
    else:
        model_inputs = rewriter_tokenizer([[kv_str, source]], truncation=True, return_tensors="pt")['input_ids']

    if (cuda):
        model_inputs = model_inputs.to('cuda')
    outputs = rewriter_model.generate(model_inputs, num_beams=5, max_length=res_max_length) # beam search
    # outputs = filler_model.generate(model_inputs, num_beams=1, do_sample=False)[0] # greedy search
    return rewriter_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def rewrite_template(
    params,
    template, 
    table,
    threshold=80,
    res_max_length=64,
    source_text=None,
    title='',
    filler_model=None,
    filler_tokenizer=None,
    rewriter_model=None,
    rewriter_tokenizer=None,
):
    pattern = re.compile(r'\#\#.+?\#\#')
    res = pattern.finditer(template)
    slots = []
    for x in res:
        start, end = x.start(), x.end()
        slots.append((start, end, template[start+2: end-2]))
    slots.sort(reverse=True)

    if (params['facts'] == 'partial'):
        candidates = list(table.keys())
        candidates = random.sample(candidates, int(len(candidates)*params['fact_ratio']))
        if (params['mode'] == 'template'):
            summary = template
            predicted_slots = []
            for slot in slots:
                start, end, query = slot
                if (len(candidates) == 0): # no data provided
                    prediction = predict_query(
                        title + ' ' + query, source_text, filler_model, filler_tokenizer
                    )
                    predicted_slots.append([query, prediction])
                else:
                    choice, score = process.extractOne(query, candidates, scorer=fuzz.token_sort_ratio)
                    # print(query)
                    # print(choice, score)
                    if (score >= threshold):
                        predicted_slots.append([query, table[choice]])
                    else:
                        prediction = predict_query(
                            query, source_text, filler_model, filler_tokenizer
                        )
                        predicted_slots.append([query, prediction])
        else:
            predicted_slots = []
            for key in candidates:
                predicted_slots.append([key, table[key]])
    else:
        predicted_slots = []
        for key in table:
            predicted_slots.append([key, table[key]])


    if (params['mode'] == 'template'):
        return rewrite(template, None, predicted_slots, res_max_length, rewriter_model, rewriter_tokenizer)
    else:
        return rewrite(None, source_text, predicted_slots, res_max_length, rewriter_model, rewriter_tokenizer)


def main(params):
    random.seed(params['seed'])

    slotsum_dataset = load_slotsum_dataset(params['data_path'], params['data_format'])
    if (params['template'] == 'predict'):
        template_set = load_predicted_templates(params['template_path'], params['data_format'])['prediction']
    else:
        template_set = load_golden_templates(slotsum_dataset, params['data_split'])
    metric = load_metric('rouge')
    data_split = params['data_split']

    assert (params['filler_path'] is not None)
    filler_tokenizer = AutoTokenizer.from_pretrained(params['filler_path'], model_max_length=params['seq_max_length'])
    filler_model = AutoModelForSeq2SeqLM.from_pretrained(params['filler_path'])

    assert (params['rewriter_path'] is not None)
    rewriter_tokenizer = AutoTokenizer.from_pretrained(params['rewriter_path'], model_max_length=params['seq_max_length'])
    rewriter_model = AutoModelForSeq2SeqLM.from_pretrained(params['rewriter_path'])

    if params['cuda']:
        filler_model.to('cuda')
        rewriter_model.to('cuda')

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
        filled_template = rewrite_template(
            params=params,
            template=template, 
            table=table, 
            source_text=source_text,
            title=title,
            threshold=params['threshold'],
            res_max_length=params['res_max_length'],
            filler_tokenizer=filler_tokenizer,
            filler_model=filler_model,
            rewriter_model=rewriter_model,
            rewriter_tokenizer=rewriter_tokenizer,
        )
        # print(template)
        # print(filled_template)
        # input()
        preds.append(filled_template)

    # save eval results
    results = compute_metrics(preds, labels)
    print(results)
    out_path = os.path.join(params['template_path'], 'eval_results_rewrite_%s_%s_%f.json' 
        %(params['mode'], params['facts'], params['fact_ratio']))
    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, indent=4)

    # save predictions for factCC
    out_path = os.path.join(params['template_path'], 'filled_predictions_rewrite_%s_%s_%f.jsonl' 
        %(params['mode'], params['facts'], params['fact_ratio']))
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
    parser.add_argument('--rewriter_path', type=str, default=None)
    parser.add_argument('--data_format', type=str, default="json")
    parser.add_argument('--data_split', type=str, default="test", choices=['test', 'validation'])
    parser.add_argument('--threshold', type=int, default=80)
    
    parser.add_argument('--mode', type=str, choices=['template', 'source'], default='template')
    parser.add_argument('--facts', type=str, choices=['partial', 'all'], default='partial')
    parser.add_argument('--fact_ratio', type=float, default=1.0)
    parser.add_argument('--template', type=str, choices=['predict', 'golden'], default='predict')
    parser.add_argument('--seq_max_length', type=int, default=512)
    parser.add_argument('--res_max_length', type=int, default=64)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    main(params)