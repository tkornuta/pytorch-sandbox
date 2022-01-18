# SPDX-License-Identifier: Apache-2.0
# based on: https://huggingface.co/blog/how-to-generate

from transformers import BertTokenizer, EncoderDecoderModel, AutoModel
from transformers import BertGenerationEncoder, GPT2LMHeadModel, BertGenerationDecoder

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Version 1: load encoder-decoder together.
#model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "gpt2")


# Version 2: load pretrained modules separatelly and join them.
encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased", bos_token_id=101, eos_token_id=102)
# add cross attention layers and use the same BOS and EOS tokens.
decoder = GPT2LMHeadModel.from_pretrained("gpt2", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

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
    num_return_sequences=5, 
    )

print(f"Output ({beam_output.shape}): {beam_output}")
print(f"Detokenized[0]: `{tokenizer.decode(beam_output[0], skip_special_tokens=False)}`")
print(f"Detokenized[0] without special tokens: `{tokenizer.decode(beam_output[0], skip_special_tokens=True)}`")
