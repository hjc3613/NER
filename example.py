from dataset.data_utils import CMeEEDataset, collate_fn_no_lavel_vec
from config.config_utils import *
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from models.BertSpanForNER import BertSpanForNER
if __name__ == '__main__':
    ds_train = CMeEEDataset('CMeEE/CMeEE_dev.json')
    dataloader = DataLoader(ds_train, batch_size=10, shuffle=False, collate_fn=collate_fn_no_lavel_vec)
    model = BertSpanForNER.from_pretrained(BERT_PATH)
    for batch in dataloader:
        batch
        outputs = model(**batch)
        outputs