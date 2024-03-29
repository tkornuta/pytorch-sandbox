# SPDX-License-Identifier: Apache-2.0
# based on: https://huggingface.co/docs/transformers/v4.15.0/en/internal/tokenization_utils#transformers.SpecialTokensMixin

from transformers import  BertTokenizerFast, BertModel, BertGenerationDecoder

# Let's see how to increase the vocabulary of Bert model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

print(tokenizer.all_special_tokens) # --> ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
print(tokenizer.all_special_ids)    # --> [100, 102, 0, 101, 103]


model = BertGenerationDecoder.from_pretrained('bert-base-uncased')

print("Original tokenizer\n"+"*"*50)
print("Vocabulary size: ", tokenizer.vocab_size)
#print("Number of special tokens: ", len(tokenizer.added_tokens_encoder)) 
print("Size of the full vocabulary with the added tokens: ", len(tokenizer)) 

# Add special tokens.
#num_added_special_toks = tokenizer.add_special_tokens({"[OBJ]":10001,"[YO]":10002})
num_added_special_toks = tokenizer.add_tokens(["[OBJ]","[YO]"], special_tokens=True)
print('We have added', num_added_special_toks, 'special tokens')

# Add "regular" tokens.
num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2', 'my_new-tok3','new_tok3'], special_tokens=False)
print('We have added', num_added_toks, 'tokens')

print("Modified tokenizer\n"+"*"*50)
print("Vocabulary size: ", tokenizer.vocab_size)
#print("Number of special tokens: ", len(tokenizer.added_tokens_encoder)) 
print("Size of the full vocabulary with the added tokens: ", len(tokenizer)) 


# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(len(tokenizer))

# Tokenize inputs and labels.
input_ids = tokenizer('This is a long article to summarize', add_special_tokens=False, return_tensors="pt").input_ids
labels = tokenizer('This is a short summary', return_tensors="pt").input_ids

# train...
output = model(input_ids=input_ids, return_dict=True)

import pdb;pdb.set_trace()
# ERROR IN THE LINE BELOW!
#greedy_output = bert2bert.generate(input_ids, max_length=50)
greedy_output = model.generate(input_ids, max_length=50)
