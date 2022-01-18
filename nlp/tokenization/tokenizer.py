# SPDX-License-Identifier: Apache-2.0

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Encode.
sent1, sent2, sent3 = "James lives in Boston", "Where is HuggingFace based?", "In New York, clearly."
encoded_dict = tokenizer(sent1, sent2)
print(encoded_dict)

# Decode.
decoded = tokenizer.decode(encoded_dict["input_ids"])
print(decoded)