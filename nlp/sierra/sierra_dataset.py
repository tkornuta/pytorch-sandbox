# SPDX-License-Identifier: Apache-2.0

import os
import csv
import h5py
from tqdm import tqdm

from torch.utils.data import Dataset

# Path to brain2.
#brain_path = "/home/tkornuta/data/brain2"

class SierraDataset(Dataset):
    """Dataset for Sierra, loading samples directly from h5 files."""

    def __init__(self, brain_path):
        # Get path to sierra data.
        sierra_path = os.path.join(brain_path, "leonardo_sierra")

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

        # Max lengths.
        self.max_plan_length = 0
        self.max_goals_length = 0

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

            #print("Symbolic Goal: ", h5["sym_goal"][()], '\n')
            self.symbolic_goals.append(h5["sym_goal"][()])

            #print("Symbolic Goal values: ", h5["sym_goal"][()], '\n')
            self.symbolic_goals_values.append(h5["sym_values"][()])

            #print("Symbolic Goal values: ", h5["sym_goal"][()], '\n')
            tokenized_goals = self.process_goals(self.symbolic_goals[-1], self.symbolic_goals_values[-1])
            self.symbolic_goals_with_negation.append(" ".join(tokenized_goals))

            #print("Symbolic Plan / Actions: ", h5["sym_plan"][()], '\n')
            plan = h5["sym_plan"][()]
            tokenized_plan = self.process_plan(plan)
            self.symbolic_plans.append(" ".join(tokenized_plan))

            # Set max lengths.
            self.max_plan_length = max(self.max_plan_length, len(tokenized_plan))
            self.max_goals_length = max(self.max_goals_length, len(tokenized_goals))


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
            "symbolic_goals": self.symbolic_goals[idx],
            "symbolic_goals_with_negation": self.symbolic_goals_with_negation[idx],
        }

        return sample
