# sort the input sentences of the dataset by tf-idf similarity

import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def sort_input_text(in_json, retain_sent_num):
    out_json = {
        'title': in_json['title'],
        # 'target_text': in_json['target_text'],
        'kv_info': in_json['kv_info'],
    }

    target_text = in_json['target_text']
    target_text = target_text.split('\n')[0]
    target_text = re.sub(r'-lrb-', '(', target_text)
    target_text = re.sub(r'-rrb-', ')', target_text)
    target_text = re.sub(r'-lsb-', '[', target_text)
    target_text = re.sub(r'-rsb-', ']', target_text)
    target_text = re.sub(r'-lcb-', '{', target_text)
    target_text = re.sub(r'-rcb-', '}', target_text)
    target_text = target_text.strip()
    out_json['target_text'] = target_text

    title_words = in_json['title'].lower().split()
    title_words = [w.strip('(').strip(')') for w in title_words]
    raw_input_text = in_json['source_text']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(raw_input_text)
    feature_names = vectorizer.get_feature_names_out()

    title_word_idx = []
    for word in title_words:
        for i, feature_name in enumerate(feature_names):
            if (feature_name == word):
                title_word_idx.append(i)

    sent_scores = []
    for i, sent in enumerate(raw_input_text):
        score = 0.0
        for idx in title_word_idx:
            score += X[i,idx]
        
        if (len(sent.split()) > 200): # bad sentence
            continue
        else:
            sent_scores.append((score, sent))

    sent_scores.sort(key=lambda x: x[0], reverse=True)
    chosen_sents = [x[1] for x in sent_scores[:retain_sent_num]]
    # out_json['source_text'] = chosen_sents
    out_json['source_text'] = ' '.join(chosen_sents)
    return out_json


if __name__ == '__main__':
    dataset_path = './data/slotsum/full'
    out_path = './data/slotsum/clipped'
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)
    splits = ['train', 'validation', 'test']

    retain_sent_num = 100

    for spl in splits:
        full_in_path = os.path.join(dataset_path, '%s.jsonl' %spl)
        full_out_path = os.path.join(out_path, '%s.jsonl' %spl)
        with open(full_in_path, 'r', encoding='utf-8') as fin:
            with open(full_out_path, 'w', encoding='utf-8') as fout:
                for line in tqdm(fin.readlines()):
                    js = json.loads(line)
                    clipped_json = sort_input_text(js, retain_sent_num)
                    json.dump(clipped_json, fout)
                    fout.write('\n')