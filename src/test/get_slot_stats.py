import os
import json
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas


if __name__ == '__main__':
    file_path = './data/slotsum/sloted'
    splits = ['train', 'validation', 'test']
    
    freq = Counter()
    example_num = 0
    for spl in splits:
        full_path = os.path.join(file_path, '%s.jsonl' %spl)
        with open(full_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                js = json.loads(line)
                slots = js['replaced_slots']
                example_num += 1
                for slot in slots:
                    slot_name = slot[0]
                    freq[slot_name] += 1
        
    sorted_freq = sorted(freq.items(), key=lambda x:x[1], reverse=True)
    # print(sorted_freq)
    tops = sorted_freq[:10]
    # print(tops)
    # print([(x[0], x[1], x[1]/example_num) for x in tops])
    top_data = [(x[0], x[1], x[1]/example_num) for x in tops]
    # for x,y,z in top_data:
    #     print(f'{x} & {y} & {round(z*100, 2)}\\% \\\\')
    # print('No. Slots: ', len(sorted_freq))

    d = {
        'rank': list(range(len(sorted_freq))),
        'frequency': [x[1] for x in sorted_freq],
    }
    df = pandas.DataFrame(data=d)
    # print(df)
    fig = sns.relplot(
        df, x='rank', y="frequency", kind="line", height=4, aspect=11.7/8.27
    )
    plt.yscale('log')
    plt.savefig('stat.pdf')