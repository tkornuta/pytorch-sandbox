# SPDX-License-Identifier: Apache-2.0
# based on:
# https://arxiv.org/pdf/1907.12461.pdf
# https://huggingface.co/docs/transformers/model_doc/bertgeneration
# https://medium.com/huggingface/encoder-decoders-in-transformers-a-hybrid-pre-trained-architecture-for-seq2seq-af4d7bf14bb8

# SPDX-License-Identifier: Apache-2.0
# based on: https://huggingface.co/docs/transformers/v4.15.0/en/internal/tokenization_utils#transformers.SpecialTokensMixin

import os
import csv
import h5py
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import  BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel
from transformers import BertTokenizer, PreTrainedTokenizerFast
from transformers import BertConfig


brain_path = "/home/tkornuta/data/brain2"
sierra_path = os.path.join(brain_path, "leonardo_sierra")
decoder_tokenizer_path = os.path.join(brain_path, "leonardo_sierra.goals_decoder_tokenizer.json")

class SierraDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, sierra_path):
        # Open csv file with commands created by humans.
        command_humans_dict = {}
        with open(os.path.join(brain_path, 'sierra_5k_v1.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                command_humans_dict[row[0][:-4]] = row[1:]

        # Prepare data structures
        self.sample_names = []
        self.command_templates = []
        self.command_humans = []
        self.symbolic_plans = []
        self.symbolic_goals = []
        self.symbolic_goals_values = []
        self.symbolic_goals_with_negation = []

        # Get files.
        sierra_files = [f for f in os.listdir(sierra_path) if os.path.isfile(os.path.join(sierra_path, f))]

        # Open files one by one.
        for i,filename in enumerate(tqdm(sierra_files)):
            # Load the file.
            h5 = h5py.File(os.path.join(sierra_path, filename), 'r')
            
            # Add sample name.
            sample_id = filename[:-3]

            # Check if sample is VALID!
            if type(command_humans_dict[sample_id]) == list and len(command_humans_dict[sample_id]) == 0:
                print(f"Skipping {i}-th sample `{sample_id}`")
                continue
            # Error for sample 4759 command: []

            self.sample_names.append(sample_id)

            # Short command generated in a scripted way.
            #print("Command (templated): ", h5["lang_description"][()], '\n')
            self.command_templates.append(h5["lang_description"][()])
            # A step-by-step plan generated in a scripted way.
            #print("Plan language: ", h5["lang_plan"][()], '\n')
            #print(command_templates[-1])

            # Human command - from another file!
            self.command_humans.append(command_humans_dict[sample_id])

            #print("Symbolic Plan / Actions: ", h5["sym_plan"][()], '\n')
            self.symbolic_plans.append(h5["sym_plan"][()])

            #print("Symbolic Goal: ", h5["sym_goal"][()], '\n')
            self.symbolic_goals.append(h5["sym_goal"][()])

            #print("Symbolic Goal values: ", h5["sym_goal"][()], '\n')
            self.symbolic_goals_values.append(h5["sym_values"][()])

            #print("Symbolic Goal values: ", h5["sym_goal"][()], '\n')
            self.symbolic_goals_with_negation.append(self.process_goals(self.symbolic_goals[-1], self.symbolic_goals_values[-1], return_string=True))

        # Make sure all lenths are the same.
        assert len(self.command_humans) == len(self.symbolic_plans)
        assert len(self.command_humans) == len(self.symbolic_goals)
        assert len(self.command_humans) == len(self.symbolic_goals_with_negation)

    def __len__(self):
        return len(self.sample_names)

    @classmethod
    def process_goals(cls, symbolic_goals, symbolic_goals_values, return_string = False):
        # Split goals and goal values.
        symbolic_goals = symbolic_goals.split("),")

        # Make sure both lists have equal number of elements.
        assert len(symbolic_goals) == len(symbolic_goals_values)

        tokenized_goals = []
        for goal, value in zip(symbolic_goals, symbolic_goals_values):
            # Add removed "),"
            if goal[-1] != ")":
                goal = goal + "),"
            # "Tokenize" goals.
            tokenized_goal = goal.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()

            # Add "not" token when required.
            if not value:
                tokenized_goal = ["not"] + tokenized_goal
            
            tokenized_goals.extend(tokenized_goal)

        if return_string:
            return " ".join(tokenized_goals)
        else:
            return tokenized_goals

    def __getitem__(self, idx):
        # Get command.
        command_humans = self.command_humans[idx]
        # For now just take first command.
        if type(command_humans) != str:
            if type(command_humans) == list and len(command_humans) > 0:
                command_humans = command_humans[0]
            else:
                print(f"Error for sample {idx} command: {command_humans}")
        sample = {
            "sample_names": self.sample_names[idx],
            "command_templates": self.command_templates[idx],
            "command_humans": command_humans,
            "symbolic_plans": self.symbolic_plans[idx],
            "symbolic_goals": self.symbolic_goals[idx],
            "symbolic_goals_with_negation": self.symbolic_goals_with_negation[idx],
        }

        return sample

# Load original BERT Ttokenizer.
encoder_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load decoder operating on the Sierra PDDL language.
decoder_tokenizer = PreTrainedTokenizerFast(tokenizer_file=decoder_tokenizer_path)
decoder_tokenizer.add_special_tokens({'unk_token': '[UNK]'})
decoder_tokenizer.add_special_tokens({'sep_token': '[SEP]'})
decoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
decoder_tokenizer.add_special_tokens({'cls_token': '[CLS]'})
decoder_tokenizer.add_special_tokens({'mask_token': '[MASK]'})
decoder_tokenizer.add_special_tokens({'bos_token': '[BOS]'})
decoder_tokenizer.add_special_tokens({'eos_token': '[EOS]'})
#print(f"\Decoder tokenizer vocabulary ({len(decoder_tokenizer.get_vocab())}):\n" + "-"*50)
#for k, v in decoder_tokenizer.get_vocab().items():
#    print(k, ": ", v)
# decoder_tokenizer.model_max_length=512 ??

# leverage checkpoints for Bert2Bert model...
# use BERT's cls token as BOS token and sep token as EOS token
encoder = BertGenerationEncoder.from_pretrained("bert-base-uncased",
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
bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)
bert2bert.config.decoder_start_token_id=decoder_tokenizer.vocab["[CLS]"]
bert2bert.config.pad_token_id=decoder_tokenizer.vocab["[PAD]"]


# Create dataset/dataloader.
sierra_ds = SierraDataset(sierra_path=sierra_path)
sierra_dl = DataLoader(sierra_ds, batch_size=256, shuffle=True, num_workers=2)

# Elementary Training.
optimizer = torch.optim.Adam(bert2bert.parameters(), lr=0.000001)
bert2bert.cuda()

for epoch in range(30):
    print("*"*50, "Epoch", epoch, "*"*50)
    for batch in tqdm(sierra_dl):
        # tokenize commands and goals.
        inputs = encoder_tokenizer(batch["command_humans"], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)
        labels = decoder_tokenizer(batch["symbolic_goals_with_negation"], return_tensors="pt", padding=True, truncation=True, add_special_tokens=True, )
        # Move to GPU.
        for key,item in inputs.items():
            if type(item).__name__ == "Tensor":
                inputs[key] = item.cuda()
        for key, item in labels.items():
            if type(item).__name__ == "Tensor":
                labels[key] = item.cuda()

        # Get outputs/loss.
        output = bert2bert(input_ids=inputs.input_ids, labels=labels.input_ids, return_dict=True)
        print("loss = ", output.loss)

        output.loss.backward()

        # Update mode.
        optimizer.step()

    print("*"*50, "Sanity check at the end of Epoch", epoch, "*"*50)
    # Sample.
    command = "Separate the given stack to form blue, red and yellow blocks stack."
    goals = "has_anything(robot),on_surface(blue_block, tabletop),stacked(blue_block, red_block),on_surface(yellow_block, tabletop)"
    values = [False, True, True, False]
    goals = SierraDataset.process_goals(goals, values, return_string=True)
    print("Command: ", command)
    print("Target: ", goals)

    # Tokenize inputs and labels.
    inputs = encoder_tokenizer(command, add_special_tokens=True, return_tensors="pt")
    print("Inputs tokenized: ", inputs)

    goals_tokenized = decoder_tokenizer(goals, add_special_tokens=True, return_tensors="pt")
    print("Target tokenized: ", goals_tokenized)
    print(f"\nTarget: `{decoder_tokenizer.decode(goals_tokenized.input_ids[0], skip_special_tokens=False)}`\n")

    # Move inputs to GPU.
    for key,item in inputs.items():
        if type(item).__name__ == "Tensor":
            inputs[key] = item.cuda()

    # Generate output:
    greedy_output = bert2bert.generate(inputs.input_ids, max_length=50)
    #print(f"Output ({greedy_output.shape}): {greedy_output}")
    print(f"\nModel prediction: `{decoder_tokenizer.decode(greedy_output[0], skip_special_tokens=False)}`\n")

