# SPDX-License-Identifier: Apache-2.0

import os
from tokenizers import Tokenizer
from copy import deepcopy

brain_path = "/home/tkornuta/data/brain2"
sierra_path = os.path.join(brain_path, "leonardo_sierra")
processed_path = os.path.join(brain_path, "processed")
decoder_tokenizer_path = os.path.join(brain_path, "leonardo_sierra.goals_decoder_tokenizer.json")

# Load the tokenizer.
tokenizer = Tokenizer.from_file(decoder_tokenizer_path)

tokens = deepcopy(tokenizer.get_vocab())
print(f"Tokenizer vocabulary ({len(tokens)}):\n" + "-"*50)
for k, v in tokens.items():
    print(k, ": ", v)

# Token types.
token_types = {
    "special": 0,
    "object": 1,
    "relation": 2, # V2: unary + binary relation distinction
    "action": 3,
    "punctuation": 4,
    "bool": 5,
    }

# Map tokens.
token_mapping = {}

while len(tokens) > 0:
    #import pdb;pdb.set_trace()
    # Get token.
    token = list(tokens.keys())[0]
    # Remove it from dict.
    tokens.pop(token)
    # Classify.
    if token in ["(", ")", ","]:
        token_mapping[token] = "punctuation" # token_types["punctuation"]

    elif token in ["true", "false", "not"]:
        token_mapping[token] = "bool"

    elif (token[0] == "<" and token[-1] == ">") or (token[0] == "[" and token[-1] == "]"):
        token_mapping[token] = "special"

    elif token in ["blue_block", "tabletop", "yellow_block", "green_block", "red_block", "robot"]:
        token_mapping[token] = "object"

    elif token in ["left", "center", "right", "far", "on_surface", "has_anything", "stacked"]:
        token_mapping[token] = "relation"

    elif any(map(token.__contains__, ["go_","stack_","place_","align_", "grasp_","lift_","release","observe_", "open_", "approach_"])):
        token_mapping[token] = "action"

    else:
        print("Not classified: ", token)


print(f"Token mapping ({len(token_mapping)}):\n"+"-"*50)
for k, v in token_mapping.items():
    print(k, ": ", v)

assert len(token_mapping) == len(tokenizer.get_vocab())

def save_to(name, list_to_save):
    filename = os.path.join(processed_path, name)
    with open(filename, "w") as f:
        for obj in list_to_save:
            if type(obj) is list:
                for item in obj:
                    f.write(item + ';')
                f.write('\n')
            else:
                f.write(obj + ';\n')

    print(f"List saved to `{filename}`")

save_to("goals_decoder_token_classes.csv", decoder_tokenizer_path)
