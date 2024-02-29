import json
import os
from datasets import load_from_disk
from transformers import BertTokenizer
from tqdm import tqdm


def inspect_wikibio(wikibio_dataset):
    print(wikibio_dataset)
    input()
    for i in range(10):
        print(wikibio_dataset['train'][i]['input_text'])
        input()
        print(wikibio_dataset['train'][i]['target_text'])
        input()
    # for entry in wikibio_dataset['train']:
    #     input_text = entry['input_text']


def inspect_wikiasp(wikiasp_dataset):
    print(wikiasp_dataset)
    input()
    # for i in range(10):
    #     print(wikiasp_dataset['train'][i]['exid'])
    #     input()
    #     print(wikiasp_dataset['train'][i]['inputs'])
    #     input()
    #     print(wikiasp_dataset['train'][i]['targets'])
    #     input()
    # key_set = set()
    # for entry in tqdm(wikiasp_dataset['train']):
    #     target = entry['targets']
    #     for pair in target:
    #         key_set.add(pair[0].lower())

    # print(key_set)

    print(wikiasp_dataset['train'][125]['exid'])
    # print(wikiasp_dataset['train'][125]['inputs'])
    print(wikiasp_dataset['train'][125]['targets'])

    print(wikiasp_dataset['train'][16178]['exid'])
    # print(wikiasp_dataset['train'][16178]['inputs'])
    print(wikiasp_dataset['train'][16178]['targets'])

    input()


def read_articles(article_path):
    # articles = []
    # with open(article_path, 'r', encoding='utf-8') as fin:
    #     for line in tqdm(fin, desc='Loading'):
    #         js = json.loads(line)
    #         articles.append(js)

    # return articles

    articles = {}
    with open(article_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin, desc='Loading'):
            js = json.loads(line)
            section_titles = sorted(list(js['article_sections'].keys()))
            cat_title = ' '.join(section_titles)
            if (cat_title not in articles):
                articles[cat_title] = []
            articles[cat_title].append(js)

    return articles


def bow_similarity(source, target):
    intersection, union = 0, 0
    for token in source:
        if not(token.isalpha() or token.startswith('##')):
            continue
        if (token in target):
            intersection += min(source[token], target[token])
            union += max(source[token], target[token])
        else:
            union += source[token]

    for token in target:
        if not(token.isalpha() or token.startswith('##')):
            continue
        if (token not in source):
            union += target[token]

    if (union == 0):
        return 1.0
    else:
        return float(intersection)/float(union)


def text_to_bow(text, tokenizer):
    freqs = {}
    # tokens = text.split()
    tokens = tokenizer.tokenize(text)
    for token in tokens:
        freqs[token] = freqs.get(token, 0)+1

    return freqs


def get_wikiasp_titles(
    domain, 
    dataset, 
    bow_articles, 
    tokenizer, 
    threshold=0.8
):
    dup = []
    
    retrieved_titles = {}
    for split in dataset:
        retrieved_titles[split] = []
        
        appeared = {}

        data = dataset[split]
        for current_idx, entry in tqdm(enumerate(data), desc='Retrieve'):
            inputs = entry['inputs']
            aspects = entry['targets']

            bow_aspects = {}
            for asp in aspects:
                section_title, section_content = asp
                bow_aspects[section_title] = text_to_bow(section_content, tokenizer)

            # remove entries without article
            empty = True
            for key in bow_aspects:
                if (len(bow_aspects[key]) > 0):
                    empty = False
                    break
            if (empty):
                continue

            bow_sections = sorted([x.lower() for x in list(bow_aspects.keys())])
            cat_bow_sections = ' '.join(bow_sections)

            # for article in bow_articles:
            for article in bow_articles.get(cat_bow_sections, []):
                flag = True
                art_sections = article['article_sections']
                title = article['title']
                title_stem = article['title_stem']

                for sect in bow_aspects:
                    if (sect.lower() not in art_sections):
                        flag = False
                        break

                    sim = bow_similarity(art_sections[sect.lower()], bow_aspects[sect])
                    if (sim < threshold):
                        flag = False
                        break

                if (flag):
                    if (title in appeared):
                        print('Already Appeared: ', title)
                        dup.append((title, split, appeared[title], current_idx))
                    else:
                        appeared[title] = current_idx
                        retrieved_titles[split].append({
                            'inputs': inputs,
                            'aspects': aspects,
                            'title': title,
                            'title_stem': title_stem,
                        })
                        print('Found title %s' %title)

                    break
            
            print('Title found: %d/%d' %(len(retrieved_titles[split]), current_idx+1))
            # if not(flag):
            #     print(aspects)
            #     input()

        print('Found titles at %s split: %d/%d' %(split, len(retrieved_titles[split]), len(data)))

    print('Duplicate titles: ')
    print(dup)
    with open('./data/slotsum/%s_duplicate_titles.txt' %domain, 'w', encoding='utf-8') as fout:
        for d in dup:
            fout.write(str(d))
            fout.write('\n')

    return retrieved_titles


if __name__ == '__main__':
    base_path = './data'

    wikibio_dataset = load_from_disk(os.path.join(base_path, 'wikibio'))
    wikiasp_dataset = {}
    wikiasp_dataset['artist'] = load_from_disk(os.path.join(base_path, 'wikiasp', 'artist'))
    wikiasp_dataset['soccer_player'] = load_from_disk(os.path.join(base_path, 'wikiasp', 'soccer_player'))

    # inspect_wikibio(wikibio_dataset)
    # inspect_wikiasp(wikiasp_dataset['soccer_player'])
    # inspect_wikiasp(wikiasp_dataset['artist'])

    article_path = './data/wikibio_articles/bow_articles.jsonl'
    wikibio_articles = read_articles(article_path)

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    domains = ['artist', 'soccer_player']
    for domain in domains:
        retrieved_titles = get_wikiasp_titles(
            domain=domain,
            dataset=wikiasp_dataset[domain],
            bow_articles=wikibio_articles,
            tokenizer=tokenizer,
            threshold=0.5,
        )

        out_path = './data/slotsum/%s' %(domain)
        if not(os.path.exists(out_path)):
            os.makedirs(out_path)
        
        for split in retrieved_titles:
            data_split = retrieved_titles[split]
            out_file = os.path.join(out_path, '%s.jsonl' %(split))
            with open(out_file, 'w', encoding='utf-8') as fout:
                for entry in data_split:
                    json.dump(entry, fout)
                    fout.write('\n')