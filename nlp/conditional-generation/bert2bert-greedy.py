# SPDX-License-Identifier: Apache-2.0
# based on:
# https://arxiv.org/pdf/1907.12461.pdf
# https://huggingface.co/docs/transformers/model_doc/bertgeneration
# https://medium.com/huggingface/encoder-decoders-in-transformers-a-hybrid-pre-trained-architecture-for-seq2seq-af4d7bf14bb8

from transformers import BertTokenizer, BertTokenizerFast, EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder


# add the EOS token as PAD token to avoid warnings
#model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

# leverage checkpoints for Bert2Bert model...
# use BERT's cls token as BOS token and sep token as EOS token
encoder = BertGenerationEncoder.from_pretrained("bert-large-uncased", bos_token_id=101, eos_token_id=102)
# add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
decoder = BertGenerationDecoder.from_pretrained("bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

# create tokenizer...
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

# Inputs.
#input_ids = tokenizer('This is a long article to summarize', add_special_tokens=False, return_tensors="pt").input_ids
#labels = tokenizer('This is a short summary', return_tensors="pt").input_ids

# train...
#loss = bert2bert(input_ids=input_ids, decoder_input_ids=labels, labels=labels).loss
#loss.backward()


# encode context the generation is conditioned on
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')

# generate text until the output length (which includes the context length) reaches 50
greedy_output = bert2bert.generate(input_ids, max_length=50)
print(f"Output ({greedy_output.shape}): {greedy_output}")
print(f"Detokenized: `{tokenizer.decode(greedy_output[0], skip_special_tokens=False)}`")

