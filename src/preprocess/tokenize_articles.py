import json
import os
from transformers import BertTokenizer
from tqdm import tqdm


if __name__ == '__main__':
    article_path = './data/wikibio_articles/refined_articles.jsonl'
    out_path = './data/wikibio_articles/bow_articles.jsonl'

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    artist_sections = [
        'career',
        'biography',
        'early life',
        'personal life',
        'music career',
        'death',
        'life and career',
        'early life and education',
        'early years',
        'exhibitions',
    ]
    soccer_player_sections = [
        'international career',
        'club career',
        'career',
        'personal life',
        'playing career',
        'early career',
        'early life',
        'professional',
        'style of play',
        'football career',
    ]

    with open(article_path, 'r', encoding='utf-8') as fin:
        with open(out_path, 'w', encoding='utf-8') as fout:
            for line in tqdm(fin):
                js = json.loads(line)
                article_sections = js['article_sections']
                bow_sections = {}
                for section_title in article_sections:
                    if (section_title.lower() in artist_sections) or (section_title.lower() in soccer_player_sections):
                        section_content = article_sections[section_title]
                        freqs = {}
                        for seg in section_content:
                            tokens = tokenizer.tokenize(seg)
                            for token in tokens:
                                freqs[token] = freqs.get(token, 0)+1

                        bow_sections[section_title.lower()] = freqs
                
                if (len(bow_sections) > 0):
                    result = {
                        'title': js['title'],
                        'title_stem': js['title_stem'],
                        'article_sections': bow_sections,
                    }
                    json.dump(result, fout, ensure_ascii=False)
                    fout.write('\n')