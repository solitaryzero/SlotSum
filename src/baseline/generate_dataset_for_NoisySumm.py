import json
import os
import nltk


if __name__ == '__main__':
    slotsum_path = './data/slotsum/clipped'
    data_splits = ['train', 'validation', 'test']

    texts = {}

    for spl in data_splits:
        texts[spl] = []
        full_path = os.path.join(slotsum_path, '%s.jsonl' %spl)
        with open(full_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                js = json.loads(line)
                title, inp, outp = js['title'], js['source_text'], js['target_text']
                texts[spl].append((title, inp, outp))

    output_path = './data/baseline_data/noisysumm'
    if not(os.path.exists(output_path)):
        os.makedirs(output_path)

    for spl in texts:
        fout = open(os.path.join(output_path, 'slotsum.%s.json' %spl), 'w', encoding='utf-8')

        for i, (title, inp, outp) in enumerate(texts[spl]):
            o = {
                'src': inp,
                'tgt': outp,
            }
            fout.write(json.dumps(o))
            fout.write('\n')

