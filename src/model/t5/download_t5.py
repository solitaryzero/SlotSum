from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
import os


def download_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    save_path = './models/pretrained/%s' %model_name.split('/')[-1]
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == "__main__":
    # download_model('t5-large')
    download_model('t5-base')