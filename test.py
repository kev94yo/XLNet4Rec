from transformers import XLNetLMHeadModel, XLNetTokenizer
import torch

model = XLNetLMHeadModel.from_pretrained('./XLNet4RecModel')
tokenizer = XLNetTokenizer('./spm/ml-1m-seq.model')

test_inputs = []
test_labels = []
with open('./data/ml-1m-seq-test.txt') as f:
    for line in f.readlines():
        test_inputs.append(line.strip())

with open('./data/ml-1m-seq-label.txt') as f:
    for line in f.readlines():
        test_labels.append(line.strip())

hits = 0
count = 0
k = 10
for i, input in enumerate(test_inputs):
    count += 1
    # input example: "1 2 3 4 <mask>"
    input_ids = torch.tensor(tokenizer.encode(input, add_special_tokens=False)).unsqueeze(0)  # We predict the masked token
    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
    perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # let's predict one token
    target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)
    outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
    next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
    # 
    index = torch.topk(next_token_logits[0][0], k = k)[1]
    for j in index:
        if tokenizer.decode(j) == test_labels[i]:
            hits += 1

print(f"HR@{k}: {hits / count}")