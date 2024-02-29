import json
import os
import re
from tqdm import tqdm
from collections import Counter
import spacy
from spaczz.matcher import FuzzyMatcher

nlp = spacy.blank("en")

def make_slots(
    in_json, 
    threshold,
    fuzzy_func='token_sort',
    debug=True,
):
    out_json = {
        'title': in_json['title'],
        'source_text': in_json['source_text'],
        'target_text': in_json['target_text'],
        'kv_info': in_json['kv_info'],
    }

    target_text = in_json['target_text']
    kv_info = in_json['kv_info']
    column_header = kv_info['column_header']
    contents = kv_info['content']
    table = [(header, content) for header, content in zip(column_header, contents)]

    if (debug):
        print('Target text: ', target_text)
        print('Table: ', str(table))

    bad_headers = [
        'article_title',
        'birth_name',
        'image', 
        'imagesize',
        'bgcolour', 
        'caption'
    ]

    doc = nlp(target_text)
    matcher = FuzzyMatcher(nlp.vocab)
    for header, content in table:
        if not(header in bad_headers):
            matcher.add(header, [nlp(content)], kwargs=[{"fuzzy_func": fuzzy_func}])
    matches = matcher(doc)

    header_index = {}
    for i, header in enumerate(column_header):
        header_index[header] = i

    all_matches = []
    for header, start, end, ratio in matches:
        # print(header, doc[start:end], ratio)
        # input()
        if (ratio >= threshold):
            all_matches.append((-ratio, header_index[header], start, end, header))

    all_matches.sort()
    selected_matches = []
    for match in all_matches:
        neg_ratio, header_index[header], start, end, header = match
        flag = True
        for s_match in selected_matches:
            _, _, s_start, s_end, _ = s_match
            if not((s_end <= start) or (s_start >= end)):
                flag = False
                break

        if (flag):
            selected_matches.append(match)
    
    for match in selected_matches:
        neg_ratio, header_index[header], start, end, header = match
        if (debug):
            print(header, doc[start:end], -neg_ratio)

    selected_matches.sort(key=lambda x: x[2], reverse=True)
    sloted_text = [t.text for t in doc]
    replaced_slots = []
    for match in selected_matches:
        neg_ratio, header_index[header], start, end, header = match
        replaced_slots.append((header, ' '.join(sloted_text[start:end])))
        sloted_text = sloted_text[:start] + ['##%s##' %header] + sloted_text[end:]

    out_json['sloted_text'] = ' '.join(sloted_text)
    out_json['replaced_slots'] = replaced_slots
    if (debug):
        print(out_json['sloted_text'])
        input()

    slot_num = len(selected_matches)
    return out_json, slot_num


if __name__ == '__main__':
    dataset_path = './data/slotsum/clipped'
    out_path = './data/slotsum/sloted'
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)
    splits = ['train', 'validation', 'test']
    slot_statistics = {}

    threshold = 80
    for spl in splits:
        full_in_path = os.path.join(dataset_path, '%s.jsonl' %spl)
        full_out_path = os.path.join(out_path, '%s.jsonl' %spl)
        slot_statistics[spl] = Counter()

        with open(full_in_path, 'r', encoding='utf-8') as fin:
            with open(full_out_path, 'w', encoding='utf-8') as fout:
                for line in tqdm(fin.readlines()):
                    js = json.loads(line)
                    sloted_json, slot_num = make_slots(
                        js, 
                        threshold=threshold,
                        fuzzy_func='token_sort',
                        debug=False,
                    )

                    json.dump(sloted_json, fout)
                    fout.write('\n')

                    slot_statistics[spl][slot_num] += 1

    recorded_statistics = {}
    for spl in slot_statistics:
        total, num_examples = 0, 0
        for x in slot_statistics[spl]:
            y = slot_statistics[spl][x]
            total += x*y
            num_examples += y

        recorded_statistics[spl] = {
            'total_slots': total,
            'record_num': num_examples,
            'avg_slots': total/num_examples,
        }

        for x in range(3):
            recorded_statistics[spl]['record_with_%d_slots' %x] = slot_statistics[spl][x]

    statistic_path = full_out_path = os.path.join(out_path, 'statistics.json')
    with open(statistic_path, 'w', encoding='utf-8') as fout:
        json.dump(recorded_statistics, fout, indent=4)