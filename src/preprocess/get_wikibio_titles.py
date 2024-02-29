from datasets import load_from_disk
import os
import re
import json
from tqdm import tqdm


def work(dataset):
    splits = ['train', 'test', 'val']
    gathered_titles = {}
    for sp in splits:
        entries = dataset[sp]
        gathered_titles[sp] = []
        for entry in tqdm(entries):
            article_title = entry['input_text']['context']
            article_title = re.sub(r'-lrb-', '(', article_title)
            article_title = re.sub(r'-rrb-', ')', article_title)
            article_title = re.sub(r'-lsb-', '[', article_title)
            article_title = re.sub(r'-rsb-', ']', article_title)
            article_title = re.sub(r'-lcb-', '{', article_title)
            article_title = re.sub(r'-rcb-', '}', article_title)
            article_title = article_title.strip()
            gathered_titles[sp].append(article_title)

    return gathered_titles


if __name__ == '__main__':
    base_path = './data'

    wikibio_dataset = load_from_disk(os.path.join(base_path, 'wikibio'))
    wikibio_titles = work(wikibio_dataset)
    
    out_path = os.path.join(base_path, 'wikibio_articles', 'wikibio_titles.json')
    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(wikibio_titles, fout, ensure_ascii=False)