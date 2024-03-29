# SPDX-License-Identifier: Apache-2.0
# based on: https://huggingface.co/docs/transformers/fast_tokenizers -> WRAPPER!!

from base64 import encode
import os
import csv
import json
from tokenizers import Tokenizer
from tokenizers import BertWordPieceTokenizer
from tokenizers.normalizers import Sequence, Replace, BertNormalizer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# Files with goals.
data_path = "/home/tkornuta/data/local-leonardo-sierra5k"
processed_path = os.path.join(data_path, "processed")
symbolic_goals = os.path.join(processed_path, "symbolic_goals.csv")
decoder_tokenizer_path = os.path.join(data_path, "leonardo_sierra.decoder_tokenizer.json")

init = True
if init:
    # Load "unified decoder vocabulary" (goals + plans).
    vocab_file = "/home/tkornuta/data/brain2/models/model_goal/dec_vocab.json"
    with open(vocab_file) as f:
        words = json.load(f)
    print(f"Loaded vocabulary ({len(words)}):\n" + "-"*50)
    for k, v in words.items():
        print(k, ": ", v)

    # Start vocabulary with all standard special tokens. (PAD=0!)
    vocab = {}
    for special_token in ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "[BOS]", "[EOS]"]:
        vocab[special_token] = len(vocab)
    
    # Copy words - but skip '<NONE>': 7, '<START>': 8, '<END>'!
    for k in words.keys():
        if k in ['<NONE>', '<START>', '<END>']:
            continue
        vocab[k] = len(vocab)
    
    # Initialize a new tokenizer with "frozen" vocabulary.
    #tokenizer = Tokenizer(BPE()) 
    #tokenizer.normalizer = Lowercase()
    #tokenizer.pre_tokenizer = CharDelimiterSplit(' ')

    init_tokenizer = BertWordPieceTokenizer(vocab=vocab) 
        #special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])  <- wrong keyword

        #bos_token = "[CLS]", eos_token = "[SEP]", unk_token = "[UNK]", sep_token = "[SEP]", <- wrong keywords
        #pad_token = "[PAD]", cls_token = "[CLS]", mask_token = "[MASK]", <- wrong keywords

    init_tokenizer.normalizer = Sequence([Replace("(", " ( "), Replace(")", " ) "), BertNormalizer()])
    init_tokenizer.pre_tokenizer = Whitespace()

    #init_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) <- must be list
    #init_tokenizer.add_special_tokens(["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "[BOS]", "[EOS]"]) <- doesn't work
    #init_tokenizer.add_special_tokens(['[PAD]']) # <- doesn't work
    #init_tokenizer.pad_token = "[PAD]" # <- doesn't work
    init_tokenizer.pad_token_id = vocab["[PAD]"]
    
    # "Set" special tokens?
    #init_tokenizer.bos_token_id = vocab["[CLS]"]
    #init_tokenizer.eos_token_id = vocab["[SEP]"]
    #init_tokenizer.unk_token_id = vocab["[UNK]"]
    #init_tokenizer.sep_token_id = vocab["[SEP]"]
    #init_tokenizer.pad_token_id = vocab["[PAD]"]
    #init_tokenizer.cls_token_id = vocab["[CLS]"]
    #init_tokenizer.mask_token_id = vocab["[MASK]"]

    # Save the created tokenizer.
    init_tokenizer.save(decoder_tokenizer_path)

    #input = "has_anything(robot),on_surface(blue_block, tabletop),stacked(blue_block, red_block),on_surface(yellow_block, tabletop)"
    #print("INPUT: ", input)
    #encoded = init_tokenizer.encode(input)
    #print(encoded) <- SPECIAL TOKENS ALREADY THERE!!

# Load the HF.tokenizers tokenizer.
#loaded_tokenizer = Tokenizer.from_file(decoder_tokenizer_path)
# "Wrap" it with HF.transformers tokenizer.
#tokenizer = PreTrainedTokenizerFast(tokenizer_object=loaded_tokenizer)

# Load from tokenizer file
tokenizer = PreTrainedTokenizerFast(tokenizer_file=decoder_tokenizer_path)
#tokenizer.add_special_tokens({'pad_token': '[PAD]'}) <- this works!
#tokenizer.pad_token

print(f"\nFinal tokenizer vocabulary ({len(tokenizer.get_vocab())}):\n" + "-"*50)
for k, v in tokenizer.get_vocab().items():
    print(k, ": ", v)

input = "has_anything(robot),on_surface(blue_block, tabletop),stacked(blue_block, red_block),on_surface(yellow_block, tabletop)"
print("INPUT: ", input)

encoded = tokenizer.encode(input) #, padding=True, truncation=True)#, return_tensors="pt")
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

compare(symbolic_goals, debug=False)
