# SPDX-License-Identifier: Apache-2.0
# based on: https://huggingface.co/blog/how-to-generate

from transformers import GPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# encode context the generation is conditioned on
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)
print(f"Output ({greedy_output.shape}): {greedy_output}")
print(f"Detokenized: `{tokenizer.decode(greedy_output[0], skip_special_tokens=False)}`")


