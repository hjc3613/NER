import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, ErnieForMaskedLM

import logging

logging.basicConfig(level=logging.INFO)  # OPTIONAL

# tokenizer = BertTokenizer.from_pretrained(r'E:\git_root\bert_models\bert-base-chinese')
# model = BertForMaskedLM.from_pretrained(r'E:\git_root\bert_models\bert-base-chinese')

tokenizer = BertTokenizer.from_pretrained(r'E:\git_root\bert_models\ernie-3.0-base-zh')
model = ErnieForMaskedLM.from_pretrained(r'E:\git_root\bert_models\ernie-3.0-base-zh')
model.eval()


# model.to('cuda')  # if you have gpu


def predict_masked_sent(text, top_k=5):
    # Tokenize input
    text = "[CLS] %s [SEP]" % text
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        print("[MASK]: '%s'" % predicted_token, " | weights:", float(token_weight))


predict_masked_sent("[MASK]骺", top_k=5)
print('############')
predict_masked_sent("小儿麻[MASK]症", top_k=5)
print('############')
predict_masked_sent("莨菪[MASK]", top_k=5)

