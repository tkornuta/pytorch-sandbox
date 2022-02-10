# SPDX-License-Identifier: Apache-2.0

import os
import csv
from tokenizers import BertWordPieceTokenizer

# Files with commands.
brain_path = "/home/tkornuta/data/brain2"
processed_path = os.path.join(brain_path, "processed")
command_templates = os.path.join(processed_path, "command_templates.csv")
command = os.path.join(processed_path, "command.csv")

# Initialize a new tokenizer
tokenizer = BertWordPieceTokenizer()

# Then train it!
tokenizer.train([ command_templates,  command], vocab_size=100)
print("Vocabulary size: ", tokenizer.get_vocab_size())
for k, v in tokenizer.get_vocab().items():
    print(k, ": ", v)

# Samples from 5k - human labels.
# data_00050000_00052798.gif,"Disjoint the given stacks to form a new stack with blue, red blocks.","Make a new stack with blue, red blocks."
# data_00150000_00150539.gif,Place all the blocks individually on the surface.,Disjoint the given stack of blocks.
# data_00110000_00110725.gif,"Separate the given stack to form yellow, red blocks stack.",Remove 2nd and 4th blocks from the given stack.
# data_00120000_00120478.gif,Remove 1st and 2nd block from the given stack and form stack with blue on top of yellow block.,Do not touch green and red block and form another stack with blue and yellow block

# Now, let's use it:
#input = "I can feel the magic, can you?"
#input = "Disjoint the given stacks to form a new stack with blue, red blocks."
#input = "Make a new stack with blue, red blocks."
input = "Remove 1st and 2nd block from the given stack and form stack with blue on top of yellow block.,Do not touch green and red block and form another stack with blue and yellow block"
print(input)
encoded = tokenizer.encode(input)#, return_tensors="pt")
print(encoded)

print(encoded.ids)
print(encoded.tokens)
print(tokenizer.decode(encoded.ids))

# Unit testing ;)
def compare(filename, debug=False):
    # Iterate through all commands.
    diffs = 0
    total = 0
    with open(filename, "r") as f:
        csvreader = csv.reader(f, delimiter=';')
        for row in csvreader:
                for command in row:
                    total += 1 
                    # "Custom" processing for comparison - remove commas and three dots.
                    command = command.replace(",", "")
                    command = command.replace("...", "")
                    command = command.replace("  ", " ")
                    command = command.replace(".", "")
                    command = command.strip()
                    # Encode and decode.
                    encoded = tokenizer.encode(command)
                    decoded = tokenizer.decode(encoded.ids)
                    if command.lower() != decoded.lower():
                        if debug:
                            print(f"{command} !=\n{decoded}")
                            import pdb;pdb.set_trace()
                        diffs += 1

    if diffs > 0:
        print(f"Decoding: DIFFERENCES for '{filename}' = {diffs} / {total}")
    else:
        print(f"Decoding: ALL {total} OK")

compare(command_templates)
compare(command)

# And finally save it somewhere
#tokenizer.save("./path/to/directory/my-bpe.tokenizer.json")