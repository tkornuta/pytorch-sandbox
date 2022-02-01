# SPDX-License-Identifier: Apache-2.0

import os
import csv
import io
import numpy as np
from PIL import Image
import logging

from dataclasses import dataclass

import h5py
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


@dataclass
class SierraDatasetConf:
    brain_path: str = "/home/tkornuta/data/brain2"
    data_subpath: str = "leonardo_sierra"
    goals_sep: bool = True
    plan_sep: bool = True
    return_rgb: bool = False
    limit: int = -1
    split: str = "train"
    split_percentage: float = 0.9

class SierraDataset(Dataset):
    """Dataset for Sierra, loading samples directly from h5 files."""

    def __init__(self, cfg: SierraDatasetConf):
        """Initializes dataset by loading humand commands (from a csv file) and all other data (from h5 files).
        
        Args:
            goals_sep (bool, default: True): if set, uses special preprocessing by removing punctuation and additional [SEP] token after each goal.
            plan_sep (bool, default: True): if set, uses special preprocessing by removing punctuation and additional [SEP] token after each action/step.
            return_rgb (bool, default: False): if set, loads and fetches images.
            limit (int, default: -1): if greater than zero, limits the number of loaded samples (mostly for testing purposes).
        """
        # Store configuration.
        self.cfg = cfg

        # Get path to sierra data.
        sierra_path = os.path.join(cfg.brain_path, cfg.data_subpath)

        # Open csv file with commands created by humans.
        command_humans_dict = {}
        with open(os.path.join(cfg.brain_path, 'sierra_5k_v1.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                command_humans_dict[row[0][:-4]] = row[1:]

        # Prepare data structures
        self.filename = []
        self.command_templates = []
        self.command_humans = []
        self.symbolic_goals = []
        self.symbolic_goals_values = []
        self.symbolic_goals_processed = []
        self.symbolic_plans = []
        self.symbolic_plans_processed = []

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
        split_file = os.path.join(cfg.brain_path, cfg.split + ".csv")
        if not os.path.exists(split_file):
            # Create all splits.
            self.split_h5_by_code()
        # Load list of files.
        sierra_files = self.get_files(split_file)
        #sierra_files = [f for f in os.listdir(sierra_path) if os.path.isfile(os.path.join(sierra_path, f))]
        
        # Open files one by one.
        for i,filename in enumerate(tqdm(sierra_files)):
            # Limit number of samples - for testing purposes.
            if cfg.limit >0 and i >= cfg.limit:
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

            self.filename.append(sample_id)

            # Short command generated in a scripted way.
            #print("Command (templated): ", h5["lang_description"][()], '\n')
            self.command_templates.append(h5["lang_description"][()])
            # A step-by-step plan generated in a scripted way.
            #print("Plan language: ", h5["lang_plan"][()], '\n')
            #print(command_templates[-1])

            # Human command - from another file!
            self.command_humans.append(command_humans_dict[sample_id])

            #print("Symbolic Goal: ", h5["sym_goal"][()], '\n')
            sym_plan = h5["sym_goal"][()]
            #print("Symbolic Goal values: ", h5["sym_goal"][()], '\n')
            sym_plan_values = h5["sym_values"][()]

            self.symbolic_goals.append(sym_plan)
            self.symbolic_goals_values.append(sym_plan_values)

            # Proces symbolic goals, depending on the settings.
            if cfg.goals_sep:
                tokenized_goals = self.process_goals_sep(sym_plan, sym_plan_values)
            else:
                tokenized_goals = self.process_goals(sym_plan, sym_plan_values)

            self.symbolic_goals_processed.append(" ".join(tokenized_goals))

            # Proces symbolic plans, depending on the settings.
            #print("Symbolic Plan / Actions: ", h5["sym_plan"][()], '\n')
            plan = h5["sym_plan"][()]
            self.symbolic_plans.append(plan)

            if cfg.goals_sep:
                tokenized_plan = self.process_plan_sep(plan)
            else:
                tokenized_plan = self.process_plan(plan)

            self.symbolic_plans_processed.append(" ".join(tokenized_plan))

            # Set max lengths.
            self.max_plan_length = max(self.max_plan_length, len(tokenized_plan))
            self.max_goals_length = max(self.max_goals_length, len(tokenized_goals))

            if cfg.return_rgb:
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
        assert len(self.command_humans) == len(self.symbolic_goals)
        assert len(self.command_humans) == len(self.symbolic_goals_processed)
        assert len(self.command_humans) == len(self.symbolic_plans)
        assert len(self.command_humans) == len(self.symbolic_plans_processed)

        print("Max plan length = ", self.max_plan_length)
        print("Max goals length = ", self.max_goals_length)
        print("Split size = ", len(self.filename))


    def split_h5_by_code(self):
        """ Split directory by code. Create a bunch of different files containing
        lists of data files used for train/val/test.

        Held-out "test" examples are determined by task code. """

        logging.info("Regenerating data splits...")

        if self.cfg.split_percentage <= 0 or self.cfg.split_percentage >= 1:
            raise RuntimeError('train val ratio must be > 0 and < 1')

        if self.cfg.split_percentage <= 0 or self.cfg.split_percentage >= 1:
            raise RuntimeError(
                'could not make ratio of ' + str(self.cfg.split_percentage) + 'work; try something between (0 - 1) range.'
            )

        train_count = int(10 * self.cfg.split_percentage)
        counter = 0

        # Get path to sierra data.
        sierra_path = os.path.join(self.cfg.brain_path, self.cfg.data_subpath)

        # Get all h5 files.
        files = os.listdir(sierra_path)
        files = [f for f in files if f.endswith('.h5')]

        # Prepare csv files.
        train_file = open(os.path.join(self.cfg.brain_path, "train.csv"), 'w')
        valid_file = open(os.path.join(self.cfg.brain_path, "valid.csv"), 'w')
        test_file = open(os.path.join(self.cfg.brain_path, "test.csv"), 'w')

        ntrain, nvalid, ntest = 0, 0, 0
        skipped = []
        for filename in tqdm(files):
            full_filename = os.path.join(sierra_path, filename)
            try:
                trial = h5py.File(full_filename, 'r')
                code = trial['task_code'][()]
                trial.close()
            except Exception as e:
                logging.error('Problem handling file: ' + str(filename))
                logging.error('Full filename: ' + str(full_filename))
                logging.error('Failed with exception: ' + str(e))
                skipped.append(filename)
                continue
            if code % 10 > 7:
                # we have a TEST example!
                test_file.writelines(filename + "\n")
                ntest += 1
            else:
                if counter < train_count:
                    train_file.writelines(filename + "\n")
                    ntrain += 1
                else:
                    valid_file.writelines(filename + "\n")
                    nvalid += 1
                counter += 1
                counter = counter % 10

        if len(skipped) > 0:
            logging.warning("Split finished with errors. Had to skip the following files: " + str(skipped))
        train_file.close()
        valid_file.close()
        test_file.close()

    def get_files(self, data_filename):
        """Returns list of filenames loaded from a csv split file."""
        files = []

        with open(data_filename, 'r') as f:
            while True:
                fname = f.readline()
                if fname is not None and len(fname) > 0:
                    files.append(fname.strip())
                else:
                    break
        return files


    def __len__(self):
        return len(self.filename)

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

    @classmethod
    def process_plan_sep(cls, symbolic_plan, return_string = False):
        """ A more sophisticated plan processing, where:
        * actions are separated by [SEP]
        * using [EOS] token after the last action
        * all punctuation (brackets, commas) are removed

        Returns:
            list of str or single string, depending on the return_string flag.
        """
        # Split goals and goal values.
        symbolic_plan = symbolic_plan.split("),")

        tokenized_plans = []
        for action in symbolic_plan:
            # Add removed "),"
            if action[-1] != ")":
                action = action + " [SEP]"

            # "Tokenize" plan into actions.
            actions = action.replace("(", " ").replace(")", " ").replace(",", " ").split()

            tokenized_plans.extend(actions)

        # Add EOS at the end.
        tokenized_plans.extend(["[EOS]"])

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
            "idx": idx,
            "filename": self.filename[idx],
            "command_templates": self.command_templates[idx],
            "command_humans": command_humans,
            "symbolic_goals": self.symbolic_goals[idx],
            "symbolic_goals_processed": self.symbolic_goals_processed[idx],
            "symbolic_plans": self.symbolic_plans[idx],
            "symbolic_plans_processed": self.symbolic_plans_processed[idx],
        }

        # Add
        if self.cfg.return_rgb:
            sample["init_rgb"] = self.init_rgb_images[idx]

        return sample

if __name__ == "__main__":
    # Create dataset/dataloader.
    sierra_cfg = SierraDatasetConf(brain_path="/home/tkornuta/data/brain2", goals_sep=True, return_rgb=True, limit=10)
    sierra_ds = SierraDataset(cfg=sierra_cfg)
    sierra_dl = DataLoader(sierra_ds, batch_size=4, shuffle=True, num_workers=2)

    print("Loaded {} samples", len(sierra_ds))
    # get sample.
    batch = next(iter(sierra_dl))
    
    import pdb;pdb.set_trace()
    print(batch.keys())