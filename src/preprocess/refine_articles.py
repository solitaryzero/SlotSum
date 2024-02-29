import os
import re
import json
from tqdm import tqdm


def extract_title(title):
    assert (title.startswith('==')) and (title.endswith('=='))
    pattern = re.compile(r'=+([^=]+)=+')
    extracted = re.sub(pattern, lambda x: x.group(1), title)
    extracted = extracted.strip()
    return extracted


def clean_text(text):
    pattern1 = re.compile(r'\[\[([^\[\]|]+)\|([^\[\]|]+)\]\]')
    pattern2 = re.compile(r'\[\[([^\[\]|]+)\]\]')
    pattern3 = re.compile(r'\[\[.+\]\]')
    text = re.sub(pattern1, lambda x: x.group(2), text)
    text = re.sub(pattern2, lambda x: x.group(1), text)
    text = re.sub(pattern3, '', text)
    return text


if __name__ == '__main__':
    raw_article_path = './data/wikibio_articles/wikibio_articles.jsonl'
    out_path = './data/wikibio_articles/refined_articles.jsonl'

    with open(raw_article_path, 'r', encoding='utf-8') as fin:
        with open(out_path, 'w', encoding='utf-8') as fout:
            for line in tqdm(fin):
                js = json.loads(line)
                title = js['title']
                title_stem = js['title_stem']
                raw_article = js['article']
                assert (raw_article.startswith('ArticleHere::;'))
                article_segs = raw_article.split('::;')[1:]
                article_sections = {}
                section = None
                for seg in article_segs:
                    seg = re.sub("'''", '', seg)
                    seg = re.sub("''", '', seg)
                    bad_chars = '&<>+-#|;'
                    for c in bad_chars:
                        seg = seg.strip(c)

                    if not(seg.startswith('==')) and ((seg[1:].startswith('=='))):
                        seg = seg[1:]

                    if not(seg.endswith('==')) and ((seg[:-1].endswith('=='))):
                        seg = seg[:-1]

                    if (seg.startswith('==')) and (seg.endswith('==')):
                        section = extract_title(seg)
                        # print(seg)
                        # print(section)
                        # input()
                        article_sections[section] = []
                    else:
                        cleaned_text = clean_text(seg)
                        # print(cleaned_text)
                        # print(seg)
                        # input()
                        if (section is None):
                            print(line)
                            print(seg)
                        article_sections[section].append(cleaned_text)

                result = {
                    'title': title,
                    'title_stem': title_stem,
                    'article_sections': article_sections,
                }
                json.dump(result, fout, ensure_ascii=False)
                fout.write('\n')