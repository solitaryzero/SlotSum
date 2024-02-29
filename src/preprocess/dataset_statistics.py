import os
import json

if __name__ == '__main__':
    dataset_path = './data/slotsum/clipped'

    stats = {
        'data_num': 0,
        'split': {
            'train': 0,
            'validation': 0,
            'test': 0,
        },
        'table': {
            'key_num': 0,
            'value_len_sum': 0,
            'avg_value_len': 0,
        },
        'source': {
            'len_sum': 0,
            'avg_len': 0,
        },
        'target': {
            'len_sum': 0,
            'avg_len': 0,
        },
    }
    splits = ['train', 'validation', 'test']

    for spl in splits:
        data_path = os.path.join(dataset_path, '%s.jsonl' %spl)

        with open(data_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                js = json.loads(line)
                stats['data_num'] += 1
                stats['split'][spl] += 1

                kv_info = js['kv_info']
                for v in kv_info['content']:
                    stats['table']['key_num'] += 1
                    val_len = len(v.split())
                    stats['table']['value_len_sum'] += val_len

                source = js['source_text']
                source_len = len(source.split())
                stats['source']['len_sum'] += source_len

                target = js['target_text']
                target_len = len(target.split())
                stats['target']['len_sum'] += target_len

    stats['table']['avg_value_len'] = stats['table']['value_len_sum'] / stats['table']['key_num']
    stats['source']['avg_len'] = stats['source']['len_sum'] / stats['data_num']
    stats['target']['avg_len'] = stats['target']['len_sum'] / stats['data_num']

    print(json.dumps(stats, indent=4))

    out_path = './data/stats'
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)

    with open(os.path.join(out_path, 'dataset_stats.json'), 'w', encoding='utf-8') as fout:
        json.dump(stats, fout, indent=4)