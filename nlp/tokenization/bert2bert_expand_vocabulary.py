# SPDX-License-Identifier: Apache-2.0
# based on: https://huggingface.co/docs/transformers/v4.15.0/en/internal/tokenization_utils#transformers.SpecialTokensMixin

import os
from transformers import  BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizer, PreTrainedTokenizerFast
from transformers import AutoConfig, BertConfig
from transformers import EncoderDecoderConfig

brain_path = "/home/tkornuta/data/brain2"
decoder_tokenizer_path = os.path.join(brain_path, "leonardo_sierra.decoder_tokenizer.json")

# Let's see how to increase the vocabulary of Bert model and tokenizer
encoder_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#decoder_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
decoder_tokenizer = PreTrainedTokenizerFast(tokenizer_file=decoder_tokenizer_path)
print(len(decoder_tokenizer))

# leverage checkpoints for Bert2Bert model...
# use BERT's cls token as BOS token and sep token as EOS token
encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased",
    bos_token_id=encoder_tokenizer.vocab["[CLS]"],
    eos_token_id=encoder_tokenizer.vocab["[SEP]"],
    )
# add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
#decoder = BertGenerationDecoder.from_pretrained("bert-base-uncased",
#    add_cross_attention=True, is_decoder=True,
#    bos_token_id=decoder_tokenizer.vocab["[CLS]"],
#    eos_token_id=decoder_tokenizer.vocab["[SEP]"],
#    )
#decoder.resize_token_embeddings(len(decoder_tokenizer))

# Fresh decoder config.
decoder_config = BertConfig(
    is_decoder = True,
    add_cross_attention = True,
    vocab_size = len(decoder_tokenizer),
    # Set required tokens.
    unk_token_id = decoder_tokenizer.vocab["[UNK]"],
    sep_token_id = decoder_tokenizer.vocab["[SEP]"],
    pad_token_id = decoder_tokenizer.vocab["[PAD]"],
    cls_token_id = decoder_tokenizer.vocab["[CLS]"],
    mask_token_id = decoder_tokenizer.vocab["[MASK]"],
    bos_token_id = decoder_tokenizer.vocab["[BOS]"],
    eos_token_id = decoder_tokenizer.vocab["[EOS]"],
    )
# AutoConfig.from_pretrained("bert-base-uncased")
#decoder_config = BertGenerationDecoderConfig()

# From: https://github.com/huggingface/transformers/blob/master/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L464
#>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
#>>> model.config.pad_token_id = tokenizer.pad_token_id
#>>> model.config.vocab_size = model.config.decoder.vocab_size
#decoder_config.decoder_start_token_id = decoder_tokenizer.vocab["[CLS]"]
# decoder_config.pad_token_type_id = 0 ?
decoder = BertGenerationDecoder(config=decoder_config)

#enc_dec_config = EncoderDecoderConfig(encoder=encoder.config, decoder=decoder.config, decoder_start_token_id=decoder_tokenizer.vocab["[CLS]"])

bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)
bert2bert.config.decoder_start_token_id=decoder_tokenizer.vocab["[CLS]"]
bert2bert.config.pad_token_id=decoder_tokenizer.vocab["[PAD]"]

# Tokenize inputs and labels.
inputs = encoder_tokenizer('Make a stack of all blocks except the green block.', add_special_tokens=False, return_tensors="pt")
print("Inputs: ", inputs)
labels = decoder_tokenizer("has_anything(robot),on_surface(blue_block, tabletop),stacked(blue_block, red_block),on_surface(yellow_block, tabletop)",
    return_tensors="pt", padding=True, truncation=True)


# train...
# v4.12.0: The decoder_input_ids are now created based on the labels, no need to pass them yourself anymore.
output = bert2bert(input_ids=inputs.input_ids, labels=labels.input_ids, return_dict=True)
print("loss = ", output.loss.shape)
output.loss.backward()

# Generate output:
greedy_output = bert2bert.generate(inputs.input_ids, max_length=50)
print(f"Output ({greedy_output.shape}): {greedy_output}")
print(f"Detokenized: `{decoder_tokenizer.decode(greedy_output[0], skip_special_tokens=False)}`")

