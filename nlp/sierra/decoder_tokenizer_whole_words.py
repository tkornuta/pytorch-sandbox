# SPDX-License-Identifier: Apache-2.0

import os
import csv
import json
from tokenizers import Tokenizer
from tokenizers import BertWordPieceTokenizer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# Files with goals.
brain_path = "/home/tkornuta/data/brain2"
processed_path = os.path.join(brain_path, "processed")
symbolic_goals = os.path.join(processed_path, "symbolic_goals.csv")
decoder_tokenizer_path = os.path.join("leonardo_sierra.decoder_tokenizer.json")

init = True
if init:
    # Load "unified decoder vocabulary" (goals + plans).
    vocab_file = "/home/tkornuta/data/brain2/models/model_goal/dec_vocab.json"
    with open(vocab_file) as f:
        vocab = json.load(f)
    print(f"Loaded vocabulary ({len(vocab)}):\n" + "-"*50)
    for k, v in vocab.items():
        print(k, ": ", v)

    # Extend vocab with the required special tokens.
    vocab["[UNK]"] = len(vocab)
    vocab["[SEP]"] = len(vocab)
    vocab["[CLS]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    vocab["[MASK]"] = len(vocab)

    # Initialize a new tokenizer with "frozen" vocabulary.
    #tokenizer = Tokenizer(BPE()) 
    #tokenizer.normalizer = Lowercase()
    #tokenizer.pre_tokenizer = CharDelimiterSplit(' ')

    init_tokenizer = BertWordPieceTokenizer(vocab=vocab)
    init_tokenizer.pre_tokenizer = Whitespace()

    # Save the created tokenizer.
    init_tokenizer.save(decoder_tokenizer_path)

# Load the HF.tokenisers tokenizer.
loaded_tokenizer = Tokenizer.from_file(decoder_tokenizer_path)
# "Wrap" it with HF.transformers tokenizer.
tokenizer = PreTrainedTokenizerFast(tokenizer_object=loaded_tokenizer)

print(f"\nFinal tokenizer vocabulary ({len(tokenizer.get_vocab())}):\n" + "-"*50)
for k, v in tokenizer.get_vocab().items():
    print(k, ": ", v)

input = "has_anything(robot),on_surface(blue_block, tabletop),stacked(blue_block, red_block),on_surface(yellow_block, tabletop)"
print("INPUT: ", input)
# Preprocessing required to the pre_tokenizer to work properly.
input = input.replace(",", " ")
print("PREPROCESSED INPUT: ", input)

encoded = tokenizer.encode(input)#, return_tensors="pt")
print(encoded)

print("DECODED: ", tokenizer.decode(encoded, skip_special_tokens=True))

# Unit testing ;)
def compare(filename, debug=False):
    # Iterate through all inputs.
    diffs = 0
    total = 0
    with open(filename, "r") as f:
        csvreader = csv.reader(f, delimiter=';')
        for row in csvreader:
                for input in row:
                    # Skip empty inputs.
                    if input == "":
                        continue
                    total += 1 
                    # Preprocessing required to the pre_tokenizer to work properly.
                    input = input.replace(",", " ")
                    # "Custom" processing for comparison - remove commas and three dots.
                    input = input.strip()
                    # Encode and decode.
                    encoded = tokenizer.encode(input)
                    # Custom postprocessing
                    input = input.replace("(", " ( ")
                    input = input.replace(")", " ) ")
                    input = input.replace("  ", " ")
                    input = input.strip()

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

compare(symbolic_goals, debug=True)
