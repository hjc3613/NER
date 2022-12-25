import json
import os
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
import torch

from .config_utils import *

class CMeEEDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        with open(path, encoding='utf8') as f:
            self.data = json.load(f)
        self.examples = [self._process_record(record) for record in self.data]

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, index):
        example = self.examples[index]
        return example
    
    def _process_record(self, record):
        text = record['text'].lower() if TO_LOWER else record['text']
        entities = record['entities']
        # 因为 [CLS], 所以索引加1，且标注规范为左闭右闭。同时训练集中有些实体边界是错的，需过滤掉
        entities = [[i['start_idx']+1, i['end_idx']+1, i['type']] for i in entities if i['start_idx'] <= i['end_idx'] and i['end_idx'] <= MAX_SEQ_LEN-2]

        inputs = TOKENIZER(list(text), is_split_into_words=True, max_length=MAX_SEQ_LEN, truncation=True)
        input_ids = inputs['input_ids']
        entity_spans = {}
        target_start = torch.zeros((len(LABELS), MAX_SEQ_LEN))
        target_end = torch.zeros((len(LABELS), MAX_SEQ_LEN))
        for start, end, label in entities:
            target_start[LABEL2IDX[label]][start] = 1
            target_end[LABEL2IDX[label]][end] = 1
            entity_spans[label] = entity_spans.get(label, []) + [(start, end, ''.join(TOKENIZER.convert_ids_to_tokens(input_ids[start:end+1])))]
        return {'inputs':inputs, 'target_start':target_start, 'target_end':target_end, 'entity_spans':entity_spans, 'text':text}
    
def collate_fn(batch):
    batch
