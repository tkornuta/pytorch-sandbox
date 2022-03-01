# SPDX-License-Identifier: Apache-2.0
# based on: https://huggingface.co/docs/transformers/fast_tokenizers -> WRAPPER!!

import os
import csv
import re
import h5py
from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer
from tokenizers.normalizers import Sequence, Replace, BertNormalizer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sierra_dataset import SierraDataset


# TOKENIZER & PROCESSING
tokenizer_name = "leonardo_sierra.goals_decoder_tokenizer_sep.json"
process_goals = SierraDataset.process_goals_sep
# Add special tokens - for decoder only!
add_special_tokens = False

# Paths.
data_path = "/home/tkornuta/data/local-leonardo-sierra5k"
sierra_path = os.path.join(data_path, "leonardo_sierra")
decoder_tokenizer_path = os.path.join(data_path, tokenizer_name)

# Get files.
sierra_files = [f for f in os.listdir(sierra_path) if os.path.isfile(os.path.join(sierra_path, f))]


init = True
if init:
    # Process goals.
    words = set()

    # Open files one by one.
    for filename in tqdm(sierra_files):
        # Load the file.
        h5 = h5py.File(os.path.join(sierra_path, filename), 'r')
        
        #print("Symbolic Goal: ", h5["sym_goal"][()], '\n')
        symbolic_goals = h5["sym_goal"][()]

        #print("Symbolic Goal: ", h5["sym_goal"][()], '\n')
        symbolic_goals_values = h5["sym_values"][()]

        # Process goals.
        tokenized_goals = process_goals(symbolic_goals, symbolic_goals_values)

        for tg in tokenized_goals:
            words.add(tg)
        
    print(f"Loaded vocabulary ({len(words)}):\n" + "-"*50)
    for v in words:
        print(v)

    # Start vocabulary with all standard special tokens. (PAD=0!)
    vocab = {}
    for special_token in ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "[BOS]", "[EOS]"]:
        vocab[special_token] = len(vocab)
    # Add other words - if not already present.
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
    print(vocab)

    # New tokenizer.
    init_tokenizer = BertWordPieceTokenizer(vocab=vocab) 
    init_tokenizer.normalizer = Sequence([Replace("(", " ( "), Replace(")", " ) "), BertNormalizer()])
    init_tokenizer.pre_tokenizer = Whitespace()
    #init_tokenizer.pad_token_id = vocab["[PAD]"]
    #print("Created tokenizer: ", init_tokenizer)

    # Save the created tokenizer.
    init_tokenizer.save(decoder_tokenizer_path)
    print("Tokenizer saved to: ", decoder_tokenizer_path)

# Load from tokenizer file.
tokenizer = PreTrainedTokenizerFast(tokenizer_file=decoder_tokenizer_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]',
    'unk_token': '[UNK]', 'mask_token': '[MASK]', 'bos_token': '[BOS]', 'eos_token': '[EOS]'
    })

print(f"\nLoaded tokenizer vocabulary ({len(tokenizer.get_vocab())}):\n" + "-"*50)
for k, v in tokenizer.get_vocab().items():
    print(k, ": ", v)

goals = "has_anything(robot),on_surface(blue_block, tabletop),stacked(blue_block, red_block),on_surface(yellow_block, tabletop)"
values = [False, True, True, False]
input = process_goals(goals, values, return_string=True)

print("-"*50)
print("INPUT: ", input)

encoded = tokenizer.encode(input, add_special_tokens=add_special_tokens) #, padding=True, truncation=True)#, return_tensors="pt")
print(encoded)

print("DECODED: ", tokenizer.decode(encoded, skip_special_tokens=False))

# Unit testing ;)
def compare(debug=False):
    # Iterate through all inputs.
    diffs = 0
    total = 0
    # Open files one by one.
    for filename in tqdm(sierra_files):
        # Load the file.
        h5 = h5py.File(os.path.join(sierra_path, filename), 'r')
        
        #print("Symbolic Goal: ", h5["sym_goal"][()], '\n')
        symbolic_goals = h5["sym_goal"][()]

        #print("Symbolic Goal: ", h5["sym_goal"][()], '\n')
        symbolic_goals_values = h5["sym_values"][()]

        # Preprocessing required to the pre_tokenizer to work properly.
        input = process_goals(symbolic_goals, symbolic_goals_values, return_string=True)

        # Encode and decode.
        encoded = tokenizer.encode(input, add_special_tokens=add_special_tokens)
        # Custom postprocessing - remove space before the comma.
        input = input.replace(" ,", ",")
        if add_special_tokens:
            input = "[CLS] " + input + " [SEP]"

        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        if input != decoded:
            if debug:
                print(f"{input} !=\n{decoded}")
                import pdb;pdb.set_trace()
            diffs += 1
        total += 1 

    if diffs > 0:
        print(f"Decoding: DIFFERENCES for '{filename}' = {diffs} / {total}")
    else:
        print(f"Decoding: ALL {total} OK")

compare(debug=True)
