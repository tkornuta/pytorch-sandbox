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

# Files with goals.
brain_path = "/home/tkornuta/data/brain2"
sierra_path = os.path.join(brain_path, "leonardo_sierra")
decoder_tokenizer_path = os.path.join(brain_path, "leonardo_sierra.goals_decoder_tokenizer.json")

def process_goals(symbolic_goals, symbolic_goals_values, return_string = False):
    # Split goals and goal values.
    symbolic_goals = symbolic_goals.split("),")

    # Make sure both lists have equal number of elements.
    assert len(symbolic_goals) == len(symbolic_goals_values)

    tokenized_goals = []
    for goal, value in zip(symbolic_goals, symbolic_goals_values):
        # Add removed "),"
        if goal[-1] != ")":
            goal = goal + "),"
        # "Tokenize" goals.
        tokenized_goal = goal.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()

        # Add "not" token when required.
        if not value:
            tokenized_goal = ["not"] + tokenized_goal
        
        tokenized_goals.extend(tokenized_goal)

    if return_string:
        return " ".join(tokenized_goals)
    else:
        return tokenized_goals


init = True
if init:
    # Get files.
    sierra_files = [f for f in os.listdir(sierra_path) if os.path.isfile(os.path.join(sierra_path, f))]

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
    for w in words:
        vocab[w] = len(vocab)

    # New tokenizer.
    init_tokenizer = BertWordPieceTokenizer(vocab=vocab) 
    init_tokenizer.normalizer = Sequence([Replace("(", " ( "), Replace(")", " ) "), BertNormalizer()])
    init_tokenizer.pre_tokenizer = Whitespace()
    init_tokenizer.pad_token_id = vocab["[PAD]"]

    # Save the created tokenizer.
    init_tokenizer.save(decoder_tokenizer_path)

# Load from tokenizer file.
tokenizer = PreTrainedTokenizerFast(tokenizer_file=decoder_tokenizer_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
import pdb;pdb.set_trace()

print(f"\Tokenizer vocabulary ({len(tokenizer.get_vocab())}):\n" + "-"*50)
for k, v in tokenizer.get_vocab().items():
    print(k, ": ", v)

goals = "has_anything(robot),on_surface(blue_block, tabletop),stacked(blue_block, red_block),on_surface(yellow_block, tabletop)"
values = [False, True, True, False]
input = process_goals(goals, values, return_string=True)

print("INPUT: ", input)

encoded = tokenizer.encode(input) #, padding=True, truncation=True)#, return_tensors="pt")
print(encoded)

print("DECODED: ", tokenizer.decode(encoded, skip_special_tokens=True))

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

        # Process goals.
        input = process_goals(symbolic_goals, symbolic_goals_values, return_string=True)

        total += 1 
        # Preprocessing required to the pre_tokenizer to work properly.
        #input = input.replace(",", " ")
        # "Custom" processing for comparison - remove commas and three dots.
        #input = input.strip()
        # Encode and decode.
        encoded = tokenizer.encode(input)
        # Custom postprocessing
        #input = input.replace("(", " ( ")
        #input = input.replace(")", " ) ")
        #input = input.replace("  ", " ")
        #input = input.strip()

        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        if input != decoded:
            if debug:
                print(f"{input} !=\n{decoded}")
                import pdb;pdb.set_trace()
            diffs += 1

    if diffs > 0:
        print(f"Decoding: DIFFERENCES for '{filename}' = {diffs} / {total}")
    else:
        print(f"Decoding: ALL {total} OK")

compare(debug=False)
