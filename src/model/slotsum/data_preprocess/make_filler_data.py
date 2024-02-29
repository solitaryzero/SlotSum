import json
import os
import re
from tqdm import tqdm
from collections import Counter


def refactor_brackets(content):
    content = re.sub(r'-lrb-', '(', content)
    content = re.sub(r'-rrb-', ')', content)
    content = re.sub(r'-lsb-', '[', content)
    content = re.sub(r'-rsb-', ']', content)
    content = re.sub(r'-lcb-', '{', content)
    content = re.sub(r'-rcb-', '}', content)

    content = re.sub(r'\( ', '(', content)
    content = re.sub(r' \)', ')', content)

    return content


def get_filler_data(
    in_json, 
    slot_names,
    debug=True,
):
    target_text = in_json['target_text']
    kv_info = in_json['kv_info']
    column_header = kv_info['column_header']
    contents = kv_info['content']
    table = [(header, content) for header, content in zip(column_header, contents)]

    if (debug):
        print('Target text: ', target_text)
        print('Table: ', str(table))

    # bad_headers = [
    #     'article_title',
    #     'birth_name',
    #     'image', 
    #     'imagesize',
    #     'bgcolour', 
    #     'caption'
    # ]

    out_json_list = []

    for header, content in table:
        # if not(header in bad_headers):
        if (header in slot_names):
            content = refactor_brackets(content)
            out_json = {
                'title': in_json['title'],
                'source_text': in_json['source_text'],
                'header': header,
                'content': content,
            }
            out_json_list.append(out_json)


    return out_json_list


if __name__ == '__main__':
    dataset_path = './data/slotsum/clipped'
    out_path = './data/slotsum/filler'
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)
    splits = ['train', 'validation', 'test']

    slot_path = './data/slotsum/sloted'
    slot_names = set()
    pattern = re.compile(r'\#\#.+?\#\#')
    for spl in splits:
        full_path = os.path.join(slot_path, '%s.jsonl' %spl)
        with open(full_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                js = json.loads(line)
                template = js['sloted_text']
                res = pattern.finditer(template)
                for x in res:
                    start, end = x.start(), x.end()
                    slot_names.add(template[start+2: end-2])

    for spl in splits:
        full_in_path = os.path.join(dataset_path, '%s.jsonl' %spl)
        full_out_path = os.path.join(out_path, '%s.jsonl' %spl)

        with open(full_in_path, 'r', encoding='utf-8') as fin:
            with open(full_out_path, 'w', encoding='utf-8') as fout:
                for line in tqdm(fin.readlines()):
                    js = json.loads(line)
                    filler_json_list = get_filler_data(
                        js,
                        slot_names=slot_names,
                        debug=False,
                    )

                    for filler_json in filler_json_list:
                        json.dump(filler_json, fout)
                        fout.write('\n')