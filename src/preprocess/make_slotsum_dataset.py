import json
import os
import re
from tqdm import tqdm
from datasets import load_from_disk


def read_wikibio(dataset):
    splits = ['train', 'test', 'val']
    title_map = {}
    for sp in splits:
        entries = dataset[sp]
        for entry in tqdm(entries):
            article_title = entry['input_text']['context']
            article_title = re.sub(r'-lrb-', '(', article_title)
            article_title = re.sub(r'-rrb-', ')', article_title)
            article_title = re.sub(r'-lsb-', '[', article_title)
            article_title = re.sub(r'-rsb-', ']', article_title)
            article_title = re.sub(r'-lcb-', '{', article_title)
            article_title = re.sub(r'-rcb-', '}', article_title)

            article_title = re.sub(r'\( ', '(', article_title)
            article_title = re.sub(r' \)', ')', article_title)
            article_title = article_title.strip()
            title_stem = article_title.lower()
            title_map[title_stem] = entry

    return title_map


def gather_data(base_path, domain, spl, wikibio_title_map):
    bad_data_path = './data/slotsum/%s_duplicate_titles.txt' %domain
    bad_titles = set()
    with open(bad_data_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tup = eval(line)
            if (tup[1] == spl):
                bad_titles.add(tup[0])

    full_path = os.path.join(base_path, domain, '%s.jsonl' %spl)

    gathered_data = []
    with open(full_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            js = json.loads(line)
            source_doc = js['inputs']
            title, title_stem = js['title'], js['title_stem']
            assert (title_stem in wikibio_title_map)
            wikibio_data = wikibio_title_map[title_stem]
            entry = {
                'title': title,
                'source_text': source_doc,
                'target_text': wikibio_data['target_text'],
                'kv_info': wikibio_data['input_text']['table'],
            }
            if (title not in bad_titles):
                gathered_data.append(entry)

    return gathered_data


if __name__ == '__main__':
    wikibio_path = './data/wikibio'

    wikibio_dataset = load_from_disk(wikibio_path)
    wikibio_title_map = read_wikibio(wikibio_dataset)

    base_path = './data/slotsum'
    domains = ['artist', 'soccer_player']
    splits = ['train', 'validation', 'test']

    all_data = {}

    for spl in splits:
        for domain in domains:
            spl_data = gather_data(base_path, domain, spl, wikibio_title_map)
            if (spl not in all_data):
                all_data[spl] = spl_data
            else:
                all_data[spl].extend(spl_data)

    out_path = './data/slotsum/full'
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)
    
    print('Train/Val/Test: %d/%d/%d' %(len(all_data['train']), len(all_data['validation']), len(all_data['test'])))

    for split in all_data:
        data_split = all_data[split]
        out_file = os.path.join(out_path, '%s.jsonl' %(split))
        with open(out_file, 'w', encoding='utf-8') as fout:
            for entry in data_split:
                json.dump(entry, fout)
                fout.write('\n')