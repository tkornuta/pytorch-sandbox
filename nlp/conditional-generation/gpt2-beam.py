# SPDX-License-Identifier: Apache-2.0
# based on: https://huggingface.co/blog/how-to-generate

from transformers import GPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# encode context the generation is conditioned on
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')

# Activate beam search and early_stopping.
# A simple remedy is to introduce n-grams (a.k.a word sequences of n words) penalties
# as introduced by Paulus et al. (2017) and Klein et al. (2017).
# The most common n-grams penalty makes sure that no n-gram appears twice by
# manually setting the probability of next words that could create an already seen n-gram to 0.
beam_output = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    early_stopping=True,
    no_repeat_ngram_size=2, 
    )

print(f"Output ({beam_output.shape}): {beam_output}")
print(f"Detokenized[0]: `{tokenizer.decode(beam_output[0], skip_special_tokens=False)}`")
