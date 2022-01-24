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
decoder_tokenizer_path = os.path.join(brain_path, "leonardo_sierra.plan_decoder_tokenizer.json")

def process_plan(symbolic_plan, return_string = False):
    # Split goals and goal values.
    symbolic_plan = symbolic_plan.split("),")

    tokenized_plans = []
    for action in symbolic_plan:
        # Add removed "),"
        if action[-1] != ")":
            action = action + "),"
        # "Tokenize" plan into actions.
        actions = action.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()

        tokenized_plans.extend(actions)

    if return_string:
        return " ".join(tokenized_plans)
    else:
        return tokenized_plans


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

        # Get plan.        
        symbolic_plan = h5["sym_plan"][()]
        
        # Process plan.
        tokenized_plans = process_plan(symbolic_plan)

        for token in tokenized_plans:
            words.add(token)
        
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

goals = "approach_obj(yellow_block),grasp_obj_on_red_block(yellow_block),lift_obj_from_red_block(yellow_block),place_on_center(yellow_block),approach_obj(red_block),grasp_obj(red_block),lift_obj_from_tabletop(red_block),align_red_block_with(blue_block),stack_red_block_on(blue_block),approach_obj(green_block),grasp_obj(green_block),lift_obj_from_far(green_block),place_on_center(green_block),approach_obj(yellow_block),grasp_obj(yellow_block),lift_obj_from_tabletop(yellow_block),align_yellow_block_with(red_block),stack_yellow_block_on(red_block),go_home(robot)"
input = process_plan(goals, return_string=True)

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
        
        # Get plan.        
        symbolic_plan = h5["sym_plan"][()]

        # Process plan.
        input = process_plan(symbolic_plan, return_string=True)

        # Encode and decode.
        encoded = tokenizer.encode(input)
        # Custom postprocessing
        input = input.replace(") ,", "),")

        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
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

compare(debug=False)
