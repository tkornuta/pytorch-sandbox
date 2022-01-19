# SPDX-License-Identifier: Apache-2.0

import os
import csv
import json
from transformers import BertTokenizer

# Files with goals.
brain_path = "/home/tkornuta/data/brain2"
processed_path = os.path.join(brain_path, "processed")
symbolic_goals = os.path.join(processed_path, "symbolic_goals.csv")

# Load "goal" vocabulary.
vocab_file = "/home/tkornuta/data/brain2/models/model_goal/dec_vocab.json"
with open(vocab_file) as f:
    vocab = json.load(f)
for k, v in vocab.items():
    print(k, ": ", v)

# Initialize pretrained tokenizer.
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Expand tokenizer vocabulary.
num_added_toks = tokenizer.add_tokens(vocab.keys())
print('We have added', num_added_toks, 'tokens')


# Now, let's use it:
input = "has_anything(robot),on_surface(blue_block, tabletop),stacked(blue_block, red_block),on_surface(yellow_block, tabletop)"
print(input)
encoded = tokenizer.encode(input)
print(encoded)
print(tokenizer.decode(encoded))

# Unit testing ;)
def compare(filename, debug=False):
    # Iterate through all commands.
    diffs = 0
    total = 0
    with open(filename, "r") as f:
        csvreader = csv.reader(f, delimiter=';')
        for row in csvreader:
                for input in row:
                    # Skip empty inputs.
                    if input == "":
                        continue
                    #print(f"input: `{input}`")
                    total += 1 
                    # "Custom" processing for comparison - remove commas and three dots.
                    input = input.replace(",", " ")
                    input = input.strip()
                    # Encode and decode.
                    encoded = tokenizer.encode(input)
                    decoded = tokenizer.decode(encoded, skip_special_tokens=True).lower()
                    if input.lower() != decoded.lower():
                        if debug:
                            print(f"{input} !=\n{decoded}")
                            #import pdb;pdb.set_trace()
                        diffs += 1

    if diffs > 0:
        print(f"Decoding is DIFFERENT for '{filename}' = {diffs} / {total}")
    else:
        print("Decoding OK")

compare(symbolic_goals, debug=False)