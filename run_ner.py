from functools import partial

from dataset.data_utils import CMeEEDataset, collate_fn_no_lavel_vec
from config.config_utils import *
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup, EvalPrediction
from models.BertSpanForNER import BertSpanForNER
from metric.metric_utils import get_ent_, cal_precision_recall_f


get_ent = partial(get_ent_, longer_best=False)

def compute_metrics(eval_pred: EvalPrediction):
    result = {}
    input_ids = eval_pred.inputs
    label_ids = eval_pred.label_ids
    mask = TOKENIZER.pad_token_id != input_ids
    origin_labels = get_ent(*label_ids, mask, input_ids)
    predictions = eval_pred.predictions
    for thresh in [0.4, 0.5, 0.6, 0.7, 0.8]:
        predictions_ = [i > thresh for i in predictions]
        pred_labels = get_ent(*predictions_, mask, input_ids)
        metric_ = cal_precision_recall_f(origin=origin_labels, pred=pred_labels)
        result.update(metric_)
    return result

def main():
    args = TrainingArguments(
        output_dir='./output',
        overwrite_output_dir=True,
        do_eval=True,
        do_train=True,
        evaluation_strategy='epoch',
        per_device_train_batch_size=10,
        num_train_epochs=5,
        # learning_rate=...
        weight_decay=0.001,
        warmup_ratio=0.15,
        lr_scheduler_type='linear',
        no_cuda=True,
        include_inputs_for_metrics=True,
        metric_for_best_model='f_0.5',
        greater_is_better=True
    )
    ds_train = CMeEEDataset('CMeEE/CMeEE_train.json')[:20]
    ds_dev = CMeEEDataset('CMeEE/CMeEE_dev.json')[:15]
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(ds_train) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(ds_train) // args.gradient_accumulation_steps * args.num_train_epochs
    model = BertSpanForNER.from_pretrained(BERT_PATH)
    

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.ernie.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in model.ernie.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': args.learning_rate},

        {"params": [p for n, p in model.start_pos_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': 0.001},
        {"params": [p for n, p in model.start_pos_classifier.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': 0.001},

        {"params": [p for n, p in model.end_pos_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': 0.001},
        {"params": [p for n, p in model.end_pos_classifier.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': 0.001},

        # {"params": [p for n, p in model.add_label_vec.named_parameters() if not any(nd in n for nd in no_decay)],
        #  "weight_decay": args.weight_decay, 'lr': 0.001},
        # {"params": [p for n, p in model.add_label_vec.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        #     , 'lr': 0.001},
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    schedule = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=int(args.warmup_ratio*t_total), num_warmup_steps=t_total)
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn_no_lavel_vec,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        optimizers=(optimizer, schedule),
        compute_metrics=compute_metrics,
        
    )

    trainer.train()
    trainer.save_model()

if __name__ == '__main__':
    main()