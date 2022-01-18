# SPDX-License-Identifier: Apache-2.0
# based on:
# https://huggingface.co/docs/transformers/model_doc/bertgeneration

from transformers import AutoTokenizer, EncoderDecoderModel

# instantiate sentence fusion model
model = EncoderDecoderModel.from_pretrained("google/roberta2roberta_L-24_discofuse")
tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")

input_ids = tokenizer('This is the first sentence. This is the second sentence.', add_special_tokens=False, return_tensors="pt").input_ids

greedy_output = model.generate(input_ids)

print(f"Output ({greedy_output.shape}): {greedy_output}")
print(f"Detokenized: `{tokenizer.decode(greedy_output[0], skip_special_tokens=False)}`")
