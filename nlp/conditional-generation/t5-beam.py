# SPDX-License-Identifier: Apache-2.0
# based on: https://huggingface.co/blog/how-to-generate

from transformers import T5ForConditionalGeneration, T5Tokenizer


tokenizer = T5Tokenizer.from_pretrained("t5-base")

# add the EOS token as PAD token to avoid warnings
model = T5ForConditionalGeneration.from_pretrained("t5-base", pad_token_id=tokenizer.eos_token_id)

# encode context the generation is conditioned on
input_ids = tokenizer.encode('Russia leader Putin is', return_tensors='pt')

# Activate beam search and early_stopping.
beam_output = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    )

print(f"Output ({beam_output.shape}): {beam_output}")
print(f"Detokenized[0]: `{tokenizer.decode(beam_output[0], skip_special_tokens=False)}`")
print(f"Detokenized[0] without special tokens: `{tokenizer.decode(beam_output[0], skip_special_tokens=True)}`")
