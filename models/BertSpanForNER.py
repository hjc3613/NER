import os
import sys
sys.path.append('.')
import pickle
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from hashlib import md5
import math

from transformers import ErnieModel, ErniePreTrainedModel, ErnieConfig
from transformers.utils import ModelOutput
import torch.nn as nn
import torch
import torch.nn.functional as F


from config.config_utils import *

def get_md5(s:str):
    return md5(s.encode('utf-8')).hexdigest()

@dataclass
class BertSpanForNEROutPut(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits_start_pos: torch.FloatTensor = None
    logits_end_pos: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class AddLabelVec(nn.Module):
    def __init__(self, cache_vecs_path, requires_grad=True):
        super().__init__()
        self.label_vecs = self._load_avg_vec_for_each_label(cache_vecs_path, requires_grad=requires_grad)

    def _load_avg_vec_for_each_label(self, path, requires_grad) -> nn.Parameter:
        if os.path.exists(path):
            with open(os.path.join(os.path.dirname(path), 'md5_of_label_in_order.txt'), encoding='utf8') as f:
                md5_of_label_order = f.read().split('\t')[0]
                assert get_md5('|'.join(LABELS)) == md5_of_label_order, "当前配置文件中LABELS与模型训练时不一致"
            with open(path, 'rb') as f:
                label_vec = pickle.load(f)
                label_vec = [label_vec[label] for label in LABELS]
                label_vec = torch.stack(label_vec)
                label_vec = label_vec.unsqueeze(1).unsqueeze(1)
                return nn.Parameter(label_vec, requires_grad=requires_grad)
        else:
            raise Exception(f"{path} not exist error, 需提前通过预训练模型{BERT_PATH}给出每个label的向量平均值，作为prompt。")
        
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        '''
        input shape: [bsz, seq_len, hid_dim]
        self.label_vecs shape: [label_num, 1, 1, hid_dim]
        output shape: [label_num, bsz, seq_len, hid_dim]
        '''
        # result == tmp
        # tmp = [input + self.label_vecs[i] for i in range(9)]
        # tmp = torch.stack(tmp)
        input = input.unsqueeze(0)
        result = input + self.label_vecs
        return result

class MultiLinear(nn.Module):
    def __init__(self, label_nums, in_features, out_features, bias: bool=True, sigmoid: bool=True) -> None:
        super(MultiLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(label_nums, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(label_nums, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.sigmoid = sigmoid
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        '''
        input shape: [bsz, label_nums, seq_len, hid_dim]
        '''
        output = torch.matmul(input, self.weight) # output shape: [bsz, label_nums, seq_len, out_features]
        if self.bias is not None:
            output += self.bias
        output = output.squeeze() # output shape: [bsz, label_nums, seq_len] because of out_features==1
        if self.sigmoid:
            output = torch.sigmoid(output)
        return output

class BertSpanForNER(ErniePreTrainedModel):
    def __init__(self, config:ErnieConfig):
        super(BertSpanForNER, self).__init__(config)
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ernie = ErnieModel(config)
        self.add_label_vec = AddLabelVec(os.path.join(BERT_PATH, F'vec_for_each_label_{BERT_NAME}.pkl'), requires_grad=False)
        self.start_pos_classifier = MultiLinear(len(LABELS), in_features=config.hidden_size, out_features=1, sigmoid=False)
        self.end_pos_classifier = MultiLinear(len(LABELS), in_features=config.hidden_size, out_features=1, sigmoid=False)

    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions_label: Optional[torch.Tensor] = None,
        end_positions_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        text: Optional[list] = None,
        entities: Optional[list]=None
    ) -> Union[Tuple[torch.Tensor], BertSpanForNEROutPut]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        seq_output_add_label_vec = self.add_label_vec(sequence_output) # shape: [label_num, bsz, seq_len, hid_dim]
        seq_output_add_label_vec = torch.transpose(seq_output_add_label_vec, 0, 1) # shape: [bsz, label_num, seq_len, hid_dim]
        logits_start_pos = self.start_pos_classifier(seq_output_add_label_vec)
        logits_end_pos = self.end_pos_classifier(seq_output_add_label_vec)

        loss = None
        if start_positions_label is not None and end_positions_label is not None:
            criterion = nn.BCEWithLogitsLoss()
            start_loss = criterion(logits_start_pos, start_positions_label)
            end_loss = criterion(logits_end_pos, end_positions_label)
            loss = start_loss + end_loss

        if not return_dict:
            output = (logits_start_pos, logits_end_pos) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return BertSpanForNEROutPut(
            loss=loss,
            logits_start_pos=torch.sigmoid(logits_start_pos),
            logits_end_pos=torch.sigmoid(logits_end_pos),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

if __name__ == '__main__':
    model = BertSpanForNER.from_pretrained(BERT_PATH)