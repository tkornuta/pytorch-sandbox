# SPDX-License-Identifier: Apache-2.0

import os
import csv
import io
from sys import breakpointhook
import numpy as np
from PIL import Image

import h5py
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SierraDataset(Dataset):
    """Dataset for Sierra, loading samples directly from h5 files."""

    def __init__(self, brain_path, goals_sep=False, return_rgb = False, limit = -1):
        """Initializes dataset by loading humand commands (from a csv file) and all other data (from h5 files).
        
        Args:
            goals_sep (bool, default: False): if set, uses special preprocessing by removing punctuation and additional [SEP] token after each goal.
            return_rgb (bool, default: False): if set, loads and fetches images.
            limit (int, default: -1): if greater than zero, limits the number of loaded samples (mostly for testing purposes).
        """
        # Get path to sierra data.
        sierra_path = os.path.join(brain_path, "leonardo_sierra")

        # Store flags.
        self.return_rgb = return_rgb

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

        self.init_rgb_images = []

        # Transforms used to reshape/normalize RGB images to format/standard used in pretrained ResNet/ViT models.
        rgb_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([224, 224]),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # Max lengths.
        self.max_plan_length = 0
        self.max_goals_length = 0

        # Get files.
        sierra_files = [f for f in os.listdir(sierra_path) if os.path.isfile(os.path.join(sierra_path, f))]
        
        # Open files one by one.
        for i,filename in enumerate(tqdm(sierra_files)):
            # Limit
            if limit >0 and i >= limit:
                break

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

            #print("Symbolic Goal: ", h5["sym_goal"][()], '\n')
            self.symbolic_goals.append(h5["sym_goal"][()])

            #print("Symbolic Goal values: ", h5["sym_goal"][()], '\n')
            self.symbolic_goals_values.append(h5["sym_values"][()])

            # Proces symbolic goals, depending on the settings.
            if goals_sep:
                tokenized_goals = self.process_goals_sep(self.symbolic_goals[-1], self.symbolic_goals_values[-1])
            else:
                tokenized_goals = self.process_goals(self.symbolic_goals[-1], self.symbolic_goals_values[-1])

            self.symbolic_goals_with_negation.append(" ".join(tokenized_goals))

            #print("Symbolic Plan / Actions: ", h5["sym_plan"][()], '\n')
            plan = h5["sym_plan"][()]
            tokenized_plan = self.process_plan(plan)
            self.symbolic_plans.append(" ".join(tokenized_plan))

            # Set max lengths.
            self.max_plan_length = max(self.max_plan_length, len(tokenized_plan))
            self.max_goals_length = max(self.max_goals_length, len(tokenized_goals))

            if self.return_rgb:
                # Number of images.
                #num_images = h5['q'].shape[0]

                # Get RGB images.
                rgbs = h5['rgb'][()]

                # Process the init RGB image.
                stream = io.BytesIO(rgbs[0])
                im = Image.open(stream)
                pil_rgb = im.convert("RGB")
                rgb_normalized_tensor = rgb_transforms(pil_rgb)
                
                self.init_rgb_images.append(rgb_normalized_tensor)

        # Make sure all lenths are the same.
        assert len(self.command_humans) == len(self.symbolic_plans)
        assert len(self.command_humans) == len(self.symbolic_goals)
        assert len(self.command_humans) == len(self.symbolic_goals_with_negation)

        print("Max plan length = ", self.max_plan_length)
        print("Max goals length = ", self.max_goals_length)

    def __len__(self):
        return len(self.sample_names)

    @classmethod
    def process_goals(cls, symbolic_goals, symbolic_goals_values, return_string = False):
        """ 
        Minimalistic goals processing, where all punctuation (brackets, commas) are kept. 
        Additionally it ads `not` token before a given predicate if the associated goal value is False.
        
        Returns:
            list of str or single string, depending on the return_string flag.
        """
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

    @classmethod
    def process_goals_sep(cls, symbolic_goals, symbolic_goals_values, return_string = False):
        """ 
        A more sophisticated goals processing, where:
        * goals are separated by [SEP]
        * using [EOS] token after the last goal
        * all punctuation (brackets, commas) are removed
        * `not` token is fused with predicate creating new "negative predicate" tokens.
        
        Returns:
            list of str or single string, depending on the return_string flag.
        """
        # Split goals and goal values.
        symbolic_goals = symbolic_goals.split("),")

        # Make sure both lists have equal number of elements.
        assert len(symbolic_goals) == len(symbolic_goals_values)

        tokenized_goals = []
        for goal, value in zip(symbolic_goals, symbolic_goals_values):
            # Add separator everywhere except at the end.
            if goal[-1] != ")":
                goal = goal + " [SEP]"

            # Fuse "not" into new token when required.
            if not value:
                goal = "not_" + goal
            
            # Remove all punctuation and split into tokens
            goals = goal.replace("(", " ").replace(")", " ").replace(",", " ").split()

            tokenized_goals.extend(goals)
        
        # Add EOS at the end.
        tokenized_goals.extend(["[EOS]"])

        if return_string:
            return " ".join(tokenized_goals)
        else:
            return tokenized_goals

    @classmethod
    def process_plan(cls, symbolic_plan, return_string = False):
        """ Minimalistic plan processing, where all punctuation (brackets, commas) are kept. 
        
        Returns:
            list of str or single string, depending on the return_string flag.
        """
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
            "symbolic_goals": self.symbolic_goals[idx],
            "symbolic_goals_with_negation": self.symbolic_goals_with_negation[idx],
        }

        # Add
        if self.return_rgb:
            sample["init_rgb"] = self.init_rgb_images[idx]

        return sample

if __name__ == "__main__":
    # Path to brain2.
    brain_path = "/home/tkornuta/data/brain2"
    # Create dataset/dataloader.
    sierra_ds = SierraDataset(brain_path=brain_path, goals_sep=True, return_rgb=True, limit=10)
    sierra_dl = DataLoader(sierra_ds, batch_size=4, shuffle=True, num_workers=2)

    print("Loaded {} samples", len(sierra_ds))
    # get sample.
    batch = next(iter(sierra_dl))
    
    import pdb;pdb.set_trace()
    print(batch.keys())