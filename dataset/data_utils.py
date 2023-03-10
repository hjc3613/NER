import json
import re
from functools import partial

from torch.utils.data import Dataset
import torch

from config.config_utils import *

def clean_text(s):
    result = re.sub(r'\ufeff', '|', s)
    return result

class CMeEEDataset(Dataset):
    def __init__(self, path_or_list) -> None:
        super().__init__()
        if isinstance(path_or_list, str):
            with open(path_or_list, encoding='utf8') as f:
                self.data = json.load(f)
        else:
            self.data = [{'text': i, 'entities': []} for i in path_or_list]
        # self.examples = [self._process_record(record) for record in self.data]

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        return example


def process_record(record):
    text = record['text'].lower() if TO_LOWER else record['text']
    text = clean_text(text)
    entities = record['entities']
    # 因为 [CLS], 所以索引加1，且标注规范为左闭右闭。同时训练集中有些实体边界是错的，需过滤掉
    entities = [[i['start_idx'] + 1, i['end_idx'] + 1, i['type']] for i in entities if
                i['start_idx'] <= i['end_idx'] and i['end_idx'] <= MAX_SEQ_LEN - 2]

    inputs = TOKENIZER(list(text), is_split_into_words=True, max_length=MAX_SEQ_LEN, truncation=True)
    input_ids = inputs['input_ids']
    entity_spans = {}
    target_start = torch.zeros((len(LABELS), MAX_SEQ_LEN))
    target_end = torch.zeros((len(LABELS), MAX_SEQ_LEN))
    for start, end, label in entities:
        target_start[LABEL2IDX[label]][start] = 1
        target_end[LABEL2IDX[label]][end] = 1
        entity_spans[label] = entity_spans.get(label, []) + [
            (start, end, ''.join(TOKENIZER.convert_ids_to_tokens(input_ids[start:end + 1])))]
    return {'inputs': inputs, 'target_start': target_start, 'target_end': target_end, 'entity_spans': entity_spans,
            'text': text}

def process_record_batch(records, has_label_vec):
    texts = [i['text'].lower() if TO_LOWER else i['text'] for i in records]
    texts = [clean_text(s) for s in texts]
    texts = [list(i) for i in texts]
    inputs = TOKENIZER(texts, is_split_into_words=True, padding="max_length", truncation=True, return_tensors='pt', max_length=MAX_SEQ_LEN)
    seq_len = inputs['input_ids'].shape[1]
    target_start = torch.zeros((len(records), len(LABELS), seq_len))
    target_end = torch.zeros((len(records), len(LABELS), seq_len))
    entity_idx_each_label = [[] for _ in range(len(records))]
    for idx, record in enumerate(records):
        entities = record['entities']
        for ent_item in entities:
            if (ent_item['start_idx'] > ent_item['end_idx']) or (ent_item['end_idx'] > MAX_SEQ_LEN - 2):
                continue
            start = ent_item['start_idx']+1
            end = ent_item['end_idx'] + 1
            entity = ent_item['entity']
            label = ent_item['type']
            entity_ = ''.join(TOKENIZER.convert_ids_to_tokens(inputs['input_ids'][idx][start:end+1]))
            if entity.lower() != entity_:
                print(entity, '->', entity_, '\ufeff' in record['text'], '\n')
            target_start[idx][LABEL2IDX[label]][start] = 1
            target_end[idx][LABEL2IDX[label]][end] = 1
            entity_idx_each_label[idx].append((start, end, label, entity))
    if has_label_vec:
        result = {**inputs, "start_positions_label":target_start, "end_positions_label":target_end, "entity_idx_each_label":entity_idx_each_label}
    else:
        result = {**inputs, "start_positions_label":target_start, "end_positions_label":target_end}
    return result


def process_record_batch2(records):
    texts = [i['text'].lower() if TO_LOWER else i['text'] for i in records]
    texts = [clean_text(s) for s in texts]
    texts = [list(i) for i in texts]
    inputs = TOKENIZER(texts, is_split_into_words=True, padding="max_length", truncation=True, return_tensors='pt', max_length=MAX_SEQ_LEN)
    seq_len = inputs['input_ids'].shape[1]
    target = torch.zeros((len(records), len(LABELS), seq_len))
    for idx, record in enumerate(records):
        entities = record['entities']
        for ent_item in entities:
            # 训练集中有少量的索引问题，例如起始位置大于结束位置，需过滤除去。大于最大句子长度的，也要过滤除去。
            if (ent_item['start_idx'] > ent_item['end_idx']) or (ent_item['end_idx'] > MAX_SEQ_LEN - 2):
                continue
            start = ent_item['start_idx']+1
            end = ent_item['end_idx'] + 1
            entity = ent_item['entity']
            label = ent_item['type']
            entity_ = ''.join(TOKENIZER.convert_ids_to_tokens(inputs['input_ids'][idx][start:end+1]))
            if entity.lower() != entity_:
                print(entity, '->', entity_)
            target[idx][LABEL2IDX[label]][start:end+1] = torch.Tensor([1] + [2]*(end-start))
    return {**inputs, 'label':target}


def collate_fn(batch, has_label_vec):
    return process_record_batch(batch, has_label_vec)

collate_fn_no_lavel_vec = partial(collate_fn, has_label_vec=False)
collate_fn_has_lavel_vec = partial(collate_fn, has_label_vec=True)
collate_fn_each_label = process_record_batch2
