from hashlib import md5

from transformers import BertTokenizer, ErnieModel

LABELS = ['sym', 'dis', 'pro', 'equ', 'dru', 'ite', 'bod', 'dep', 'mic']
LABEL2IDX = {label:idx for idx, label in enumerate(LABELS)}
IDX2LABEL = {idx:label for label, idx in LABEL2IDX.items()}

BERT_NAME = "ernie-3.0-base-zh"
BERT_PATH = f'W:/bert-models/{BERT_NAME}'

DEVICE = 'cuda:0'

TOKENIZER:BertTokenizer = BertTokenizer.from_pretrained(BERT_PATH)
# ENCODER = ErnieModel.from_pretrained(BERT_PATH)

MAX_SEQ_LEN = 512

TO_LOWER = True
