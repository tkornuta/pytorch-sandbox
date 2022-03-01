# SPDX-License-Identifier: Apache-2.0

from base64 import encode
import os
import csv
import json
from tqdm import tqdm
from transformers import BertTokenizer

# Files with goals.
data_path = "/home/tkornuta/data/local-leonardo-sierra5k"
processed_path = os.path.join(data_path, "processed")
symbolic_goals = os.path.join(processed_path, "symbolic_goals.csv")

# Load "unified decoder vocabulary" (goals + goalss).
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
    min_goals_length = 100000
    avg_goals_length = 0
    max_goals_length = 0

    # Iterate through all commands.
    diffs = 0
    total = 0
    with open(filename, "r") as f:
        csvreader = csv.reader(f, delimiter=';')
        for row in tqdm(csvreader):
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

                    # Statistics.
                    len_goals = len(encoded)
                    min_goals_length = min(min_goals_length, len_goals)
                    avg_goals_length += len_goals
                    max_goals_length = max(max_goals_length, len_goals)

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

    # Show basic split statistics.
    avg_goals_length = int(avg_goals_length / total)
    print(f"Number of tokens in goals | Min = {min_goals_length} | Avg = {avg_goals_length} | Max = {max_goals_length}")

compare(symbolic_goals, debug=False)