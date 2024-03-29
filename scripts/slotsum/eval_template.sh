CUDA_VISIBLE_DEVICES=5 python ./src/model/slotsum/eval_template.py \
    --data_path ./data/slotsum/sloted \
    --model_path ./models/slotsum/template/bart-large \
    --base_model ./models/pretrained/bart-large \
    --data_format json \
    --seq_max_length 512 \
    --cuda \
    --learning_rate 1e-5 \
    --epoch 4 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --print_interval 100 \
    --seed 42