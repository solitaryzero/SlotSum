import os
import argparse
import torch
import random
import time
import numpy as np
import re
import nltk

# import wandb
# wandb.init(project="slotsum-pegasus")

from tqdm import tqdm, trange

from transformers import (
    AutoTokenizer,
    PegasusForConditionalGeneration, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from datasets import load_dataset, load_metric


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


def main(params):
    tokenizer = AutoTokenizer.from_pretrained(params['base_model'], model_max_length=params['seq_max_length'])
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    def tokenize_function(examples):
        source_text = examples['source_text']
        target_text = examples['target_text']

        if (params['additional_data'] == 'kv'):
            kv_info = examples['kv_info']
            kvs = []
            for x in kv_info:
                headers = x['column_header']
                contents = x['content']
                t = []
                for key, value in zip(headers, contents):
                    t.append('%s | %s' %(key, value))

                kvs.append(' # '.join(t))

            inputs = []
            for s, k in zip(source_text, kvs):
                inputs.append([k, s])
            model_inputs = tokenizer(inputs, padding='max_length', truncation=True)
        elif (params['additional_data'] == 'key'):
            kv_info = examples['kv_info']
            keys = []
            for x in kv_info:
                headers = x['column_header']
                t = ' | '.join(headers)

                keys.append(t)

            inputs = []
            for s, k in zip(source_text, keys):
                inputs.append([k, s])
            model_inputs = tokenizer(inputs, padding='max_length', truncation=True)
        else:
            model_inputs = tokenizer(source_text, padding='max_length', truncation=True)

        labels = tokenizer(target_text, padding='max_length', truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    full_dataset = load_slotsum_dataset(params['data_path'], params['data_format'])
    tokenized_dataset = full_dataset.map(tokenize_function, batched=True)

    metric = load_metric('rouge')
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}


    args = Seq2SeqTrainingArguments(
        params['output_path'],
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=params['print_interval'],
        save_strategy="epoch",
        learning_rate=params['learning_rate'],
        per_device_train_batch_size=params['train_batch_size'],
        per_device_eval_batch_size=params['eval_batch_size'],
        weight_decay=0.01,
        save_total_limit=4,
        num_train_epochs=params['epoch'],
        predict_with_generate=True,
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        # report_to="wandb"
        report_to="none"
    )

    def init_model():
        model = PegasusForConditionalGeneration.from_pretrained(params['base_model'])
        if (params['cuda']):
            model = model.to('cuda')
        return model

    trainer = Seq2SeqTrainer(
        model_init=init_model,
        args=args,
        train_dataset=tokenized_dataset['train'].shuffle(seed=42),
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_state()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--data_path', type=str, default='./data/slotsum/clipped')
    parser.add_argument('--output_path', type=str, default='./models/pegasus')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--base_model', type=str, default="google/pegasus-xsum")
    parser.add_argument('--data_format', type=str, default="json")
    
    # model arguments
    parser.add_argument('--seq_max_length', type=int, default=512)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--additional_data', type=str, choices=['none', 'key', 'kv'], default='none')

    # training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    main(params)