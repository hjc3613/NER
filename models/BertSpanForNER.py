import os
import pickle

from transformers import ErnieModel, BertPreTrainedModel, ErniePreTrainedModel
import torch.nn as nn
from torch.nn.modules import Module

from config.config_utils import *

class BertSpanForNER(ErniePreTrainedModel):
    def __init__(self, config):
        super(BertSpanForNER, self).__init__(config)
        self.config = config
        self.ernie = ErnieModel(config)
        self._load_avg_vec_for_each_label(os.path.join(config.name_or_path, 'vec_for_each_label_erine3.pkl'))

    def _load_avg_vec_for_each_label(self, path):
        with open(path, 'rb') as f:
            label_vec = pickle.load(f)
        LABEL2IDX

if __name__ == '__main__':
    model = BertSpanForNER.from_pretrained(r"E:\git_root\bert_models\ernie-3.0-base-zh")