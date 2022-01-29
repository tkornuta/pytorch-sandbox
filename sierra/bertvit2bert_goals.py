# SPDX-License-Identifier: Apache-2.0
# based on:
# https://arxiv.org/pdf/1907.12461.pdf
# https://huggingface.co/docs/transformers/model_doc/bertgeneration
# https://medium.com/huggingface/encoder-decoders-in-transformers-a-hybrid-pre-trained-architecture-for-seq2seq-af4d7bf14bb8
# based on: https://huggingface.co/docs/transformers/v4.15.0/en/internal/tokenization_utils#transformers.SpecialTokensMixin

import os
import csv
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import  BertGenerationEncoder, BertGenerationDecoder
from transformers import BertTokenizer, PreTrainedTokenizerFast
from transformers import BertConfig
from transformers import ViTModel  # ViTForImageClassification, ViTConfig

from multimodal_encoder_decoder import MultimodalEncoderDecoderModel
from sierra_dataset import SierraDataset

# TOKENIZER & PROCESSING
tokenizer_name = "leonardo_sierra.goals_decoder_tokenizer_sep.json"
limit=-1

# Paths.
brain_path = "/home/tkornuta/data/brain2"
sierra_path = os.path.join(brain_path, "leonardo_sierra")
decoder_tokenizer_path = os.path.join(brain_path, tokenizer_name)

# Load original BERT Ttokenizer.
encoder_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load decoder operating on the Sierra PDDL language.
decoder_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=decoder_tokenizer_path,
    unk_token = '[UNK]',
    sep_token = '[SEP]',
    pad_token = '[PAD]',
    cls_token = '[CLS]]',
    mask_token = '[MASK]',
)
decoder_tokenizer.add_special_tokens({'bos_token': '[BOS]','eos_token': '[EOS]'})
#print(f"\Decoder tokenizer vocabulary ({len(decoder_tokenizer.get_vocab())}):\n" + "-"*50)
#for k, v in decoder_tokenizer.get_vocab().items():
#    print(k, ": ", v)
# decoder_tokenizer.model_max_length=512 ??

# Create dataset/dataloader.
sierra_ds = SierraDataset(brain_path=brain_path, goals_sep=True, return_rgb=True, limit=limit)
sierra_dl = DataLoader(sierra_ds, batch_size=64, shuffle=True, num_workers=2)

# Create ViT encoder .
image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
#torch.nn.Linear(self.vit.config.hidden_size, cfg.n_classes)

# leverage checkpoints for Bert2Bert model...
# use BERT's cls token as BOS token and sep token as EOS token
command_encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased",
    # Set required tokens.
    #bos_token_id=encoder_tokenizer.vocab["[CLS]"],
    #eos_token_id=encoder_tokenizer.vocab["[SEP]"],
    )

# Fresh decoder config.
decoder_config = BertConfig(
    is_decoder = True,
    add_cross_attention = True, # add cross attention layers
    vocab_size = len(decoder_tokenizer),
    # Set required tokens.
    unk_token_id = decoder_tokenizer.vocab["[UNK]"],
    sep_token_id = decoder_tokenizer.vocab["[SEP]"],
    pad_token_id = decoder_tokenizer.vocab["[PAD]"],
    cls_token_id = decoder_tokenizer.vocab["[CLS]"],
    mask_token_id = decoder_tokenizer.vocab["[MASK]"],
    bos_token_id = decoder_tokenizer.vocab["[BOS]"],
    eos_token_id = decoder_tokenizer.vocab["[EOS]"],
    )
# Initialize a brand new bert-based decoder.
decoder = BertGenerationDecoder(config=decoder_config)

# Setup enc-decoder mode.
model = MultimodalEncoderDecoderModel(image_encoder=image_encoder, command_encoder=command_encoder, decoder=decoder)
model.config.decoder_start_token_id=decoder_tokenizer.vocab["[CLS]"]
model.config.pad_token_id=decoder_tokenizer.vocab["[PAD]"]

# Move model to GPU.
model.cuda()

# Sample one item from the dataset.
print("Loaded {} samples", len(sierra_ds))
# Get sample.
sample = next(iter(DataLoader(sierra_ds, batch_size=1, shuffle=True)))
#sample = sierra_ds[14]

# Sample.
print(f"Sample {sample['idx']}: {sample['sample_names']}\n" + "-"*100)
print("Command: ", sample["command_humans"])
print("Target: ", sample["symbolic_goals_with_negation"])
print("-"*100)

input_image = sample["init_rgb"]
# Tokenize inputs.
command_tokenized = encoder_tokenizer(sample["command_humans"], add_special_tokens=True, return_tensors="pt")
print("Command tokenized: ", command_tokenized)
print(f"\nCommand: `{encoder_tokenizer.decode(command_tokenized.input_ids[0], skip_special_tokens=False)}`\n")

# Tokenize labels - do not add special tokens as they are already due to the preprocessing.
target_tokenized = decoder_tokenizer(sample["symbolic_goals_with_negation"], add_special_tokens=False, return_tensors="pt") 
print("Target tokenized: ", target_tokenized)
print(f"\nTarget: `{decoder_tokenizer.decode(target_tokenized.input_ids[0], skip_special_tokens=False)}`\n")

# Move data to GPU.
input_image = input_image.cuda()
for key,item in command_tokenized.items():
    if type(item).__name__ == "Tensor":
        command_tokenized[key] = item.cuda()
for key,item in target_tokenized.items():
    if type(item).__name__ == "Tensor":
        target_tokenized[key] = item.cuda()

# Elementary training.
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
for epoch in range(30):
    print("*"*50, "Epoch", epoch, "*"*50)
    for batch in tqdm(sierra_dl):
        # tokenize commands and goals.
        commands = encoder_tokenizer(batch["command_humans"], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)
        labels = decoder_tokenizer(batch["symbolic_goals_with_negation"], add_special_tokens=False, return_tensors="pt", padding=True, truncation=True, )
        
        # Move data to GPU.
        for key,item in commands.items():
            if type(item).__name__ == "Tensor":
                commands[key] = item.cuda()
        for key, item in labels.items():
            if type(item).__name__ == "Tensor":
                labels[key] = item.cuda()

        batch["init_rgb"] = batch["init_rgb"].cuda()

        # Get outputs/loss.
        output = model(input_image=batch["init_rgb"], input_ids=commands.input_ids, labels=labels.input_ids, return_dict=True)
        print("loss = ", output.loss)

        output.loss.backward()

        # Update mode.
        optimizer.step()

    print("*"*50, "Sanity check at the end of Epoch", epoch, "*"*50)
    print("Command tokenized: ", command_tokenized)
    print(f"\nCommand: `{encoder_tokenizer.decode(command_tokenized.input_ids[0], skip_special_tokens=False)}`\n")

    print("Target tokenized: ", target_tokenized)
    print(f"\nTarget: `{decoder_tokenizer.decode(target_tokenized.input_ids[0], skip_special_tokens=False)}`\n")

    # Generate output:
    greedy_output = model.generate(input_image=input_image, input_ids=command_tokenized.input_ids, max_length=(sierra_ds.max_goals_length + 2))
    print(f"Output ({greedy_output.shape}): {greedy_output}")
    print(f"\nModel prediction: `{decoder_tokenizer.decode(greedy_output[0], skip_special_tokens=False)}`\n")

