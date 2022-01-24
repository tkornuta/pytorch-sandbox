# SPDX-License-Identifier: Apache-2.0
# based on:
# https://arxiv.org/pdf/1907.12461.pdf
# https://huggingface.co/docs/transformers/model_doc/bertgeneration
# https://medium.com/huggingface/encoder-decoders-in-transformers-a-hybrid-pre-trained-architecture-for-seq2seq-af4d7bf14bb8
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
decoder_tokenizer_path = os.path.join(brain_path, "leonardo_sierra.plan_decoder_tokenizer.json")

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
        self.max_plan_length = 0

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
            plan = h5["sym_plan"][()]
            tokenized_plan = self.process_plan(plan)
            self.symbolic_plans.append(" ".join(tokenized_plan))

            # Set max length.
            self.max_plan_length = max(self.max_plan_length, len(tokenized_plan))

        print("Max plan length = ", self.max_plan_length)
        # Make sure all lenths are the same.
        assert len(self.command_humans) == len(self.symbolic_plans)

    def __len__(self):
        return len(self.sample_names)

    @classmethod
    def process_plan(cls, symbolic_plan, return_string = False):
        # Split goals and goal values.
        symbolic_plan = symbolic_plan.split("),")

        tokenized_plans = []
        for action in symbolic_plan:
            # Add removed "),"
            if action[-1] != ")":
                action = action + "),"
            # "Tokenize" plan into actions.
            actions = action.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()

            tokenized_plans.extend(actions)

        if return_string:
            return " ".join(tokenized_plans)
        else:
            return tokenized_plans

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
            #"symbolic_goals": self.symbolic_goals[idx],
            #"symbolic_goals_with_negation": self.symbolic_goals_with_negation[idx],
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

# Create dataset/dataloader.
sierra_ds = SierraDataset(sierra_path=sierra_path)
sierra_dl = DataLoader(sierra_ds, batch_size=64, shuffle=True, num_workers=2)

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

# Elementary Training.
optimizer = torch.optim.Adam(bert2bert.parameters(), lr=0.000001)
bert2bert.cuda()

for epoch in range(30):
    print("*"*50, "Epoch", epoch, "*"*50)
    if True:
        for batch in tqdm(sierra_dl):
            # tokenize commands and goals.
            inputs = encoder_tokenizer(batch["command_humans"], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)
            labels = decoder_tokenizer(batch["symbolic_plans"], return_tensors="pt", padding=True, max_length=sierra_ds.max_plan_length, truncation=True, add_special_tokens=True, )

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
    orig_plan = "approach_obj(yellow_block),grasp_obj_on_red_block(yellow_block),lift_obj_from_red_block(yellow_block),place_on_center(yellow_block),approach_obj(red_block),grasp_obj(red_block),lift_obj_from_tabletop(red_block),align_red_block_with(blue_block),stack_red_block_on(blue_block),approach_obj(green_block),grasp_obj(green_block),lift_obj_from_far(green_block),place_on_center(green_block),approach_obj(yellow_block),grasp_obj(yellow_block),lift_obj_from_tabletop(yellow_block),align_yellow_block_with(red_block),stack_yellow_block_on(red_block),go_home(robot)"
    plan = SierraDataset.process_plan(orig_plan, return_string=True)
    #action = SierraDataset.process_plans(plan, return_string=True)
    print("Command: ", command)
    print("Target: ", plan)

    # Tokenize inputs and labels.
    inputs = encoder_tokenizer(command, add_special_tokens=True, return_tensors="pt")
    print("Inputs tokenized: ", inputs)

    plan_tokenized = decoder_tokenizer(plan, add_special_tokens=True, return_tensors="pt")
    print("Target tokenized: ", plan_tokenized)
    print(f"\nTarget: `{decoder_tokenizer.decode(plan_tokenized.input_ids[0], skip_special_tokens=False)}`\n")

    # Move inputs to GPU.
    for key,item in inputs.items():
        if type(item).__name__ == "Tensor":
            inputs[key] = item.cuda()

    # Generate output:
    greedy_output = bert2bert.generate(inputs.input_ids, max_length=200)
    #print(f"Output ({greedy_output.shape}): {greedy_output}")
    print(f"\nModel prediction: `{decoder_tokenizer.decode(greedy_output[0], skip_special_tokens=False)}`\n")

