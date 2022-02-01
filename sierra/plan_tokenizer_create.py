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

# Paths.
brain_path = "/home/tkornuta/data/brain2"
sierra_path = os.path.join(brain_path, "leonardo_sierra")

# Tokenizer settings.
decoder_tokenizer_path = os.path.join(brain_path, "leonardo_sierra.plan_decoder_tokenizer_sep.json")
process_goals = SierraDataset.process_plan_sep
add_special_tokens = False

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

        # Get plan.        
        symbolic_plan = h5["sym_plan"][()]
        
        # Process plan.
        tokenized_plans = process_goals(symbolic_plan)

        for token in tokenized_plans:
            words.add(token)
        
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
    # init_tokenizer.pad_token_id = vocab["[PAD]"]

    # Save the created tokenizer.
    init_tokenizer.save(decoder_tokenizer_path)
    print("\nTokenizer saved to: ", decoder_tokenizer_path)

# Load from tokenizer file.
tokenizer = PreTrainedTokenizerFast(tokenizer_file=decoder_tokenizer_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]',
    'unk_token': '[UNK]', 'mask_token': '[MASK]', 'bos_token': '[BOS]', 'eos_token': '[EOS]'
    })

print(f"\nLoaded tokenizer vocabulary ({len(tokenizer.get_vocab())}):\n" + "-"*50)
for k, v in tokenizer.get_vocab().items():
    print(k, ": ", v)

goals = "approach_obj(yellow_block),grasp_obj_on_red_block(yellow_block),lift_obj_from_red_block(yellow_block),place_on_center(yellow_block),approach_obj(red_block),grasp_obj(red_block),lift_obj_from_tabletop(red_block),align_red_block_with(blue_block),stack_red_block_on(blue_block),approach_obj(green_block),grasp_obj(green_block),lift_obj_from_far(green_block),place_on_center(green_block),approach_obj(yellow_block),grasp_obj(yellow_block),lift_obj_from_tabletop(yellow_block),align_yellow_block_with(red_block),stack_yellow_block_on(red_block),go_home(robot)"
input = process_goals(goals, return_string=True)

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
        
        # Get plan.        
        symbolic_plan = h5["sym_plan"][()]

        # Process plan.
        input = process_goals(symbolic_plan, return_string=True)

        # Encode and decode.
        encoded = tokenizer.encode(input, add_special_tokens=add_special_tokens)
        # Custom postprocessing - remove space before the comma.
        input = input.replace(" ,", ",")

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

compare(debug=False)
