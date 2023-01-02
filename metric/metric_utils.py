import sys
sys.path.append('.')

from typing import Tuple
from tqdm import tqdm
import re

from sklearn.metrics import classification_report as sklearn_class_report
import numpy as np

from config.config_utils import *

def get_ent_(start_pos, end_pos, mask, input_ids, longer_best):
    '''
    shape: [bsz, label_nums, seq_len]
    '''
    result = []
    for sent_idx, (start_tensor, end_tensor, mask_, sentence_ids) in tqdm(enumerate(zip(start_pos, end_pos, mask, input_ids)), desc=f'decode', total=len(start_pos)):
        for label_idx in range(len(LABELS)):
            type_ = IDX2LABEL[label_idx]
            start_idx_lst = [idx for idx, val in enumerate(start_tensor[label_idx][:sum(mask_)]) if val==1]
            end_idx_lst = [idx for idx, val in enumerate(end_tensor[label_idx][:sum(mask_)]) if val==1]
            cur_end = None
            for start in start_idx_lst:
                # 如果有多个实体共享同一个end，有两种策略：短实体优先、长实体优先
                if cur_end is not None and longer_best and start < end:
                    continue
                for end in end_idx_lst:
                    if end >= start:
                        entity = ''.join(TOKENIZER.convert_ids_to_tokens(sentence_ids[start:end+1]))
                        result.append((sent_idx, start, end, type_, entity))
                        cur_end = end
                        break
    return result

def get_ent2_(label_ids, mask, input_ids):
    result = []
    for sample_idx in tqdm(range(len(label_ids)), desc='对label进行解码', total=len(label_ids)):
        seq_len = sum(mask[sample_idx])
        token_ids = input_ids[sample_idx][:seq_len]
        for label_idx in range(len(LABELS)):
            label = ''.join(str(i) for i in label_ids[sample_idx][label_idx][:seq_len])
            type_ = IDX2LABEL[label_idx]
            entities = [(sample_idx, i.start(), i.end(), ''.join(TOKENIZER.convert_ids_to_tokens(token_ids[i.start():i.end()])), type_) for i in re.finditer(r'12*', label)]
            result.extend((entities))
    return result

def cal_precision_recall_f(origin, pred, thresh):
    print(f'######################### {thresh} ##############################\n')
    print(f'ground true\n: {origin[:30]} \n...\n {origin[-30:]}')
    print(f'predict result\n: {pred[:30]} \n...\n {pred[-30:]}')
    intersection = set(origin).intersection(set(pred))
    if len(intersection) == 0:
        precision = 0
        recall = 0
        f = 0
    else:
        precision = len(intersection) / len(set(pred))
        recall = len(intersection) / len(set(origin))
        f = 2*precision*recall/(precision+recall)
    result = {f'precision_{thresh}':precision, f'recall_{thresh}':recall, f'f_{thresh}':f}
    return result

def classification_report(pred:np.ndarray, origin:np.ndarray, mask:np.ndarray):
    '''
    shape: [bsz, label_num, seq_len]
    '''
    pred_ = {}
    origin_ = {}
    # 遍历每个example
    for pred_pos, origin_pos, mask_ in tqdm(zip(pred, origin, mask), desc='decode position0/1', total=len(pred)):
        seq_len = sum(mask_)
        # 遍历每个label
        for label_idx, (pred_seq, origin_seq) in enumerate(zip(pred_pos, origin_pos)):
            label_name = IDX2LABEL[label_idx]
            pred_seq = list(pred_seq)[:seq_len]
            origin_seq = list(origin_seq)[:seq_len]
            pred_[label_name] = pred_.get(label_name, []) + pred_seq
            origin_[label_name] = origin_.get(label_name, []) + origin_seq

    for label_name in LABELS:
        pred_label = pred_[label_name]
        origin_label = origin_[label_name]
        print(f'############################## {label_name} ##############################')
        sklearn_class_report(y_pred=pred_label, y_true=origin_label)