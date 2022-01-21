# SPDX-License-Identifier: Apache-2.0
# based on:
# https://arxiv.org/pdf/1907.12461.pdf
# https://huggingface.co/docs/transformers/model_doc/bertgeneration
# https://medium.com/huggingface/encoder-decoders-in-transformers-a-hybrid-pre-trained-architecture-for-seq2seq-af4d7bf14bb8

# SPDX-License-Identifier: Apache-2.0
# based on: https://huggingface.co/docs/transformers/v4.15.0/en/internal/tokenization_utils#transformers.SpecialTokensMixin

import os

from transformers import  BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel
from transformers import BertTokenizer, PreTrainedTokenizerFast
from transformers import BertConfig

brain_path = "/home/tkornuta/data/brain2"
decoder_tokenizer_path = os.path.join(brain_path, "leonardo_sierra.decoder_tokenizer.json")

# Load original BERT Ttokenizer.
encoder_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load decoder operating on the Sierra PDDL language.
decoder_tokenizer = PreTrainedTokenizerFast(tokenizer_file=decoder_tokenizer_path)

# leverage checkpoints for Bert2Bert model...
# use BERT's cls token as BOS token and sep token as EOS token
encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased",
    # Set required tokens.
    #bos_token_id=encoder_tokenizer.vocab["[CLS]"],
    #eos_token_id=encoder_tokenizer.vocab["[SEP]"],
    )

# Fresh decoder config.
decoder_config = BertConfig(
    is_decoder = True,
    add_cross_attention = True, # add cross attention layers
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
# Initialize a brand new bert-based decoder.
decoder = BertGenerationDecoder(config=decoder_config)

# Setup enc-decoder mode.
bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)
bert2bert.config.decoder_start_token_id=decoder_tokenizer.vocab["[CLS]"]
bert2bert.config.pad_token_id=decoder_tokenizer.vocab["[PAD]"]

# Sample.
command = "Separate the given stack to form blue, red and yellow blocks stack."
goals = "has_anything(robot),on_surface(blue_block, tabletop),stacked(blue_block, red_block),on_surface(yellow_block, tabletop)"

# Tokenize inputs and labels.
inputs = encoder_tokenizer(command, add_special_tokens=False, return_tensors="pt")
print("Inputs: ", inputs)
labels = decoder_tokenizer(goals, return_tensors="pt")

# train...
# v4.12.0: The decoder_input_ids are now created based on the labels, no need to pass them yourself anymore.
output = bert2bert(input_ids=inputs.input_ids, labels=labels.input_ids, return_dict=True)
print("loss = ", output.loss.shape)
output.loss.backward()

# Generate output:
greedy_output = bert2bert.generate(inputs.input_ids, max_length=50)
print(f"Output ({greedy_output.shape}): {greedy_output}")
print(f"Detokenized: `{decoder_tokenizer.decode(greedy_output[0], skip_special_tokens=False)}`")
