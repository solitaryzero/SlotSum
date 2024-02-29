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
                source_sents = nltk.tokenize.sent_tokenize(inp)
                inp = ' <EOP> '.join(source_sents)
                target_sents = nltk.tokenize.sent_tokenize(outp)
                outp = ' <SNT> '.join(target_sents)
                texts[spl].append((title, inp, outp))

    output_path = './data/baseline_data/wikicatsum'
    if not(os.path.exists(output_path)):
        os.makedirs(output_path)

    for spl in texts:
        fout_src = open(os.path.join(output_path, '%s.src' %spl), 'w', encoding='utf-8')
        fout_tgt = open(os.path.join(output_path, '%s.tgt' %spl), 'w', encoding='utf-8')

        for i, (title, inp, outp) in enumerate(texts[spl]):
            fout_src.write('%s <EOT> %s\n' %(title, inp))
            fout_tgt.write('%s\n' %(outp))

        fout_src.close()
        fout_tgt.close()
