from collections import defaultdict
from tqdm import tqdm
import pickle
import os

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from config.config_utils import *
from transformers import ErnieModel
from dataset.data_utils import CMeEEDataset, collate_fn_has_lavel_vec

@torch.no_grad()
def get_label_vecs(path):
    model = ErnieModel.from_pretrained(BERT_PATH)
    model = model.to(DEVICE)
    dataset = CMeEEDataset(r'CMeEE/CMeEE_train.json')
    dataloder = DataLoader(dataset, collate_fn=collate_fn_has_lavel_vec, shuffle=False, batch_size=16)
    vecs_each_label = {}
    vecs_each_label_num = {}

    for batch in tqdm(dataloder, desc='获取每个标签的向量'):
        entity_idx_each_label = batch.pop('entity_idx_each_label')
        start_positions = batch.pop('start_positions_label')
        end_positions = batch.pop('end_positions_label')
        batch = {k:v.to(DEVICE) for k,v in batch.items()}
        outputs = model(**batch)
        seq_output = outputs.last_hidden_state
        seq_output
        for idx, entities in enumerate(entity_idx_each_label):
            hidden_state = seq_output[idx]
            for start, end, label, entity in entities:
                ent_vec = hidden_state[start:end+1]
                ent_vec = torch.mean(ent_vec, 0)
                vecs_each_label[label] = vecs_each_label.get(label, torch.zeros(ent_vec.shape[0])) + ent_vec
                vecs_each_label_num[label] = vecs_each_label_num.get(label, 0) + 1
    
    print(vecs_each_label_num)
    vecs_each_label = {k:v/vecs_each_label_num[k] for k,v in vecs_each_label.items()}
    vecs_each_label = {k:v.to('cpu') for k,v in vecs_each_label.items()}
    with open(path, mode='wb') as f:
        pickle.dump(vecs_each_label, f)


if __name__ == '__main__':
    get_label_vecs(os.path.join(BERT_PATH, F'vec_for_each_label_{BERT_NAME}.pkl'))