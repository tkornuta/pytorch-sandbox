# SPDX-License-Identifier: Apache-2.0

from base64 import encode
import os
import csv
import json
from tqdm import tqdm
from transformers import BertTokenizer

# Files with plan.
data_path = "/home/tkornuta/data/local-leonardo-sierra5k"
processed_path = os.path.join(data_path, "processed")
symbolic_plan = os.path.join(processed_path, "symbolic_plans.csv")

# Load "unified decoder vocabulary" (plan + goals).
vocab_file = "/home/tkornuta/data/brain2/models/model_goal/dec_vocab.json"
with open(vocab_file) as f:
    vocab = json.load(f)
for k, v in vocab.items():
    print(k, ": ", v)

# Initialize pretrained tokenizer.
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Expand tokenizer vocabulary.
num_added_toks = tokenizer.add_tokens(vocab.keys())
print('We have added', num_added_toks, 'tokens\n', "-"*50)

# Now, let's use it:
input = "approach_obj(yellow_block),grasp_obj_on_red_block(yellow_block),lift_obj_from_red_block(yellow_block),place_on_center(yellow_block),approach_obj(red_block),grasp_obj(red_block),lift_obj_from_tabletop(red_block),align_red_block_with(blue_block),stack_red_block_on(blue_block),approach_obj(green_block),grasp_obj(green_block),lift_obj_from_far(green_block),place_on_center(green_block),approach_obj(yellow_block),grasp_obj(yellow_block),lift_obj_from_tabletop(yellow_block),align_yellow_block_with(red_block),stack_yellow_block_on(red_block),go_home(robot)"
num_actions = len(input.split("),"))
print(f"Input (number of actions: {num_actions}): {input}\n")
encoded = tokenizer.encode(input)
print(f"Tokenized input (number of plan tokens {len(encoded)}): {encoded}\n")
print("Detokenized: ", tokenizer.decode(encoded))

# Unit testing ;)
def compare(filename, debug=False):
    min_plan_length = 100000
    avg_plan_length = 0
    max_plan_length = 0

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
                    len_plan = len(encoded)
                    min_plan_length = min(min_plan_length, len_plan)
                    avg_plan_length += len_plan
                    max_plan_length = max(max_plan_length, len_plan)

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
    avg_plan_length = int(avg_plan_length / total)
    print(f"Number of tokens in plan | Min = {min_plan_length} | Avg = {avg_plan_length} | Max = {max_plan_length}")

compare(symbolic_plan, debug=False)