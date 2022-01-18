# SPDX-License-Identifier: Apache-2.0

from transformers import BertTokenizer
import json

# Get all words from output vocabulary.
#vocab_file = "/home/tkornuta/data/brain2/models/model_goal/enc_vocab.json"
vocab_file = "/home/tkornuta/data/brain2/models/model_goal/dec_vocab.json"
with open(vocab_file) as f:
    vocab = json.load(f)
for k, v in vocab.items():
    print(k, ": ", v)

# Create a tokenizer.
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Expand vocabulary.
tokenizer.add_tokens(vocab.keys())

# Encode.
sent1 = " ".join(list(vocab.keys())[3:])
print(f"INPUT: ({len(sent1.split())}): {sent1}")

encoded_dict = tokenizer(sent1)
#print(encoded_dict)



# Decode.
decoded = tokenizer.decode(encoded_dict["input_ids"], skip_special_tokens=True)
print(f"DECODED: ({len(decoded.split())}): {decoded}")

#assert tokenizer.decode(encoded_dict["input_ids"], skip_special_tokens=True) == sent1