import os
import pickle
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from transformers import ErnieModel, BertPreTrainedModel, ErniePreTrainedModel
from transformers.utils import ModelOutput
import torch.nn as nn
from torch.nn.modules import Module
import torch


from config.config_utils import *

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
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class BertSpanForNER(ErniePreTrainedModel):
    def __init__(self, config):
        super(BertSpanForNER, self).__init__(config)
        self.config = config
        self.ernie = ErnieModel(config)
        self._load_avg_vec_for_each_label(os.path.join(config.name_or_path, F'vec_for_each_label_{BERT_PATH}.pkl'))

    def _load_avg_vec_for_each_label(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                label_vec = pickle.load(f)
        else:
            raise Exception(f"{path} not exist error, 需提前通过预训练模型{BERT_PATH}给出每个label的向量平均值，作为prompt。")

    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        entity_idx_each_label:Optional[dict] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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

        logits = ...

        loss = None
        if start_positions is not None and end_positions is not None:
            # calc loss
            ...
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return BertSpanForNEROutPut(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

if __name__ == '__main__':
    model = BertSpanForNER.from_pretrained(r"E:\git_root\bert_models\ernie-3.0-base-zh")