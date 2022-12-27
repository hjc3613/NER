from transformers import ErnieModel, BertPreTrainedModel
import torch.nn as nn
from torch.nn.modules import Module

from config.config_utils import *

class BertSpanForNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSpanForNER, self).__init__(config)
        self.bert = ErnieModel