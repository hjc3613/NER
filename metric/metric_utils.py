import sys
sys.path.append('.')
from config.config_utils import *

def get_ent_(start_pos, end_pos, mask, input_ids, longer_best):
    '''
    shape: [bsz, label_nums, seq_len]
    '''
    result = []
    for sent_idx, start_tensor, end_tensor, mask_, sentence_ids in enumerate(zip(start_pos, end_pos, mask, input_ids)):
        for label_idx in range(len(LABELS)):
            type_ = IDX2LABEL[label_idx]
            start_idx_lst = [idx for idx, val in enumerate(start_tensor[label_idx][1:sum(mask_)]) if val==1]
            end_idx_lst = [idx for idx, val in enumerate(end_tensor[label_idx][1:sum(mask_)]) if val==1]
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
    return result

def cal_precision_recall_f(origin, pred, thresh):
    intersection = set(origin).intersection(set(pred))
    precision = len(intersection) / len(set(pred))
    recall = len(intersection) / len(set(origin))
    f = 2*precision*recall/(precision+recall)
    result = {f'precision_{thresh}':precision, f'recall_{thresh}':recall, f'f_{thresh}':f}
    return result