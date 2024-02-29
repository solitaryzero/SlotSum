import json
import os


if __name__ == '__main__':
    sloted_data_file = './data/slotsum/sloted/test.jsonl'
    with open(sloted_data_file, 'r', encoding='utf-8') as fin:
        all_data = []
        for i, line in enumerate(fin):
            js = json.loads(line)
            all_data.append(js)

    prediction_base_path = './data/slotsum/predictions'
    prediction_paths = {
        'BART-raw': '%s/bart_raw/prediction.jsonl' %prediction_base_path,
        # 'BART-cnn': '%s/bart_cnn/prediction.jsonl' %prediction_base_path,
        'T5': '%s/t5-base/prediction.jsonl' %prediction_base_path,
        'Pegasus': '%s/pegasus/prediction.jsonl' %prediction_base_path,
        'NoisySumm': '%s/noisysumm/prediction.jsonl' %prediction_base_path,
        'SlotSum_predict': '%s/template_prediction_nocnn/filled_predictions_predict.jsonl' %prediction_base_path,
        'SlotSum_allpredict': '%s/template_prediction_nocnn/filled_predictions_allpredict.jsonl' %prediction_base_path,
        'SlotSum_discard': '%s/template_prediction_nocnn/filled_predictions_discard.jsonl' %prediction_base_path,
        # 'SlotSum_keep': '%s/template_prediction_nocnn/filled_predictions_keep.jsonl' %prediction_base_path,
    }

    # model = 'NoisySumm'
    model = 'SlotSum_discard'
    with open(prediction_paths[model], 'r', encoding='utf-8') as fin:
        all_predictions = []
        for i, line in enumerate(fin):
            js = json.loads(line)
            all_predictions.append(js)

    entry_index = 507

    dat = all_data[entry_index]
    title = dat['title']
    target_text = dat['target_text']
    print(title)
    print('Golden: ')
    print(target_text)
    print('Template: ')
    print(dat['sloted_text'])
    print('Prediction: ')
    print(all_predictions[entry_index]['claim'])
    input()
    print('Input: ')
    print(dat['source_text'])