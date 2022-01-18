# SPDX-License-Identifier: Apache-2.0
# based on:
# https://huggingface.co/blog/encoder-decoder

from transformers import MarianMTModel, MarianTokenizer

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")

# create ids of encoded input vectors
input_ids = tokenizer("I want to buy a car", return_tensors="pt").input_ids

# translate example
output_ids = model.generate(input_ids)[0]

# decode and print
print(tokenizer.decode(output_ids))
