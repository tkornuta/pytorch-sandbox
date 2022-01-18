# SPDX-License-Identifier: Apache-2.0
# based on: https://huggingface.co/blog/how-to-generate

from transformers import GPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# encode context the generation is conditioned on
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')

# activate sampling and deactivate top_k by setting top_k sampling to 0
# use temperature to decrease the sensitivity to low probability candidates
sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=50, 
    top_k=0,
    temperature=0.7,
    )

print(f"Output ({sample_output.shape}): {sample_output}")
print(f"Detokenized: `{tokenizer.decode(sample_output[0], skip_special_tokens=False)}`")


