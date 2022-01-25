# SPDX-License-Identifier: Apache-2.0
# based on:
# https://arxiv.org/pdf/1907.12461.pdf
# https://huggingface.co/docs/transformers/model_doc/bertgeneration
# https://medium.com/huggingface/encoder-decoders-in-transformers-a-hybrid-pre-trained-architecture-for-seq2seq-af4d7bf14bb8
# based on: https://huggingface.co/docs/transformers/v4.15.0/en/internal/tokenization_utils#transformers.SpecialTokensMixin

import os
import csv
import h5py
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import  BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel
from transformers import BertTokenizer, PreTrainedTokenizerFast
from transformers import BertConfig

from sierra_dataset import SierraDataset

# TOKENIZER & PROCESSING
tokenizer_name = "leonardo_sierra.goals_decoder_tokenizer_sep.json"
process_goals = SierraDataset.process_goals_sep
# For dataset.
goals_sep = True
# Add special tokens - for decoder only!
add_special_tokens = False

# Paths.
brain_path = "/home/tkornuta/data/brain2"
sierra_path = os.path.join(brain_path, "leonardo_sierra")
decoder_tokenizer_path = os.path.join(brain_path, tokenizer_name)

# Load original BERT Ttokenizer.
encoder_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load decoder operating on the Sierra PDDL language.
decoder_tokenizer = PreTrainedTokenizerFast(tokenizer_file=decoder_tokenizer_path)
decoder_tokenizer.add_special_tokens({'unk_token': '[UNK]'})
decoder_tokenizer.add_special_tokens({'sep_token': '[SEP]'})
decoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
decoder_tokenizer.add_special_tokens({'cls_token': '[CLS]'})
decoder_tokenizer.add_special_tokens({'mask_token': '[MASK]'})
decoder_tokenizer.add_special_tokens({'bos_token': '[BOS]'})
decoder_tokenizer.add_special_tokens({'eos_token': '[EOS]'})
#print(f"\Decoder tokenizer vocabulary ({len(decoder_tokenizer.get_vocab())}):\n" + "-"*50)
#for k, v in decoder_tokenizer.get_vocab().items():
#    print(k, ": ", v)
# decoder_tokenizer.model_max_length=512 ??

# Create dataset/dataloader.
sierra_ds = SierraDataset(brain_path=brain_path, goals_sep=goals_sep)
sierra_dl = DataLoader(sierra_ds, batch_size=256, shuffle=True, num_workers=2)

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

# Elementary Training.
optimizer = torch.optim.Adam(bert2bert.parameters(), lr=0.000001)
bert2bert.cuda()

for epoch in range(30):
    print("*"*50, "Epoch", epoch, "*"*50)
    for batch in tqdm(sierra_dl):
        # tokenize commands and goals.
        inputs = encoder_tokenizer(batch["command_humans"], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)
        labels = decoder_tokenizer(batch["symbolic_goals_with_negation"], return_tensors="pt", padding=True, truncation=True, add_special_tokens=add_special_tokens)
        # Move to GPU.
        for key,item in inputs.items():
            if type(item).__name__ == "Tensor":
                inputs[key] = item.cuda()
        for key, item in labels.items():
            if type(item).__name__ == "Tensor":
                labels[key] = item.cuda()

        # Get outputs/loss.
        output = bert2bert(input_ids=inputs.input_ids, labels=labels.input_ids, return_dict=True)
        print("loss = ", output.loss)

        output.loss.backward()

        # Update mode.
        optimizer.step()

    print("*"*50, "Sanity check at the end of Epoch", epoch, "*"*50)
    # Sample.
    command = "Separate the given stack to form blue, red and yellow blocks stack."
    goals = "has_anything(robot),on_surface(blue_block, tabletop),stacked(blue_block, red_block),on_surface(yellow_block, tabletop)"
    values = [False, True, True, False]
    goals = process_goals(goals, values, return_string=True)
    print("Command: ", command)
    print("Target: ", goals)

    # Tokenize inputs and labels.
    inputs = encoder_tokenizer(command, add_special_tokens=True, return_tensors="pt")
    print("Inputs tokenized: ", inputs)

    goals_tokenized = decoder_tokenizer(goals, add_special_tokens=add_special_tokens, return_tensors="pt")
    print("Target tokenized: ", goals_tokenized)
    print(f"\nTarget: `{decoder_tokenizer.decode(goals_tokenized.input_ids[0], skip_special_tokens=False)}`\n")

    # Move inputs to GPU.
    for key,item in inputs.items():
        if type(item).__name__ == "Tensor":
            inputs[key] = item.cuda()

    # Generate output:
    greedy_output = bert2bert.generate(inputs.input_ids, max_length=(sierra_ds.max_goals_length + 2))
    #print(f"Output ({greedy_output.shape}): {greedy_output}")
    print(f"\nModel prediction: `{decoder_tokenizer.decode(greedy_output[0], skip_special_tokens=False)}`\n")

