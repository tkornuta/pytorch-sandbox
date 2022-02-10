# SPDX-License-Identifier: Apache-2.0

import os
import csv
import io
from typing import List
import numpy as np
from PIL import Image
import logging

from dataclasses import dataclass, field

import h5py
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


@dataclass
class SierraDatasetConf:
    brain_path: str = "/home/tkornuta/data/brain2"
    data_subpath: str = "leonardo_sierra"
    process_goals: str = "sep" # Options: clean, sep
    process_plans: str = "split" # Options: clean, sep, split
    skip_actions: List = field(default_factory=lambda: ["approach", "go_home"]) # List of actions to be skipped.
    add_pad: bool = False
    return_rgb: bool = False
    limit: int = -1
    split: str = "train"
    split_percentage: float = 0.9

class SierraDataset(Dataset):
    """Dataset for Sierra, loading samples directly from h5 files."""

    def __init__(self, cfg: SierraDatasetConf):
        """Initializes dataset by loading humand commands (from a csv file) and all other data (from h5 files).
        
        Args:
            process_goals (str, default: "sep"): goals processing, default: remove punctuation, add additional [SEP] token after each goal.
            process_plans (str, default: "split"): plans processing, default: remove punctuation, split verb, add additional [SEP] token after each action/step.
            add_pad (bool, default: False): if set, adds a single, additional [PAD] at the end of each goals/plan sequence.
            return_rgb (bool, default: False): if set, loads and fetches images.
            limit (int, default: -1): if greater than zero, limits the number of loaded samples (mostly for testing purposes).
        """
        # Store configuration.
        self.cfg = cfg

        # Get path to sierra data.
        sierra_path = os.path.join(cfg.brain_path, cfg.data_subpath)

        # Open csv file with commands created by humans.
        command_from_humans_dict = {}
        with open(os.path.join(cfg.brain_path, 'sierra_5k_v1.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                command_from_humans_dict[row[0][:-4]] = row[1:]

        # Prepare data structures
        self.filename = []
        self.command_from_templates = []
        self.command_from_goals_templates = []
        self.command_from_humans = []
        self.symbolic_goals = []
        self.symbolic_goals_values = []
        self.symbolic_goals_processed = []
        self.symbolic_plan = []
        self.symbolic_plan_processed = []
        self.symbolic_plan_skipped_actions = [] # List of lists.

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
        self.min_goals_length = 0
        self.avg_goals_length = 0
        self.max_goals_length = 0
        self.min_plan_length = 0
        self.avg_plan_length = 0
        self.max_plan_length = 0

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
            if type(command_from_humans_dict[sample_id]) == list and len(command_from_humans_dict[sample_id]) == 0:
                logging.warning(f"Skipping {i}-th sample `{sample_id}`")
                continue
            # Error for sample 4759 command: []

            self.filename.append(sample_id)

            # Human command - from another file!
            self.command_from_humans.append(command_from_humans_dict[sample_id])
            # Short command generated in a scripted way.
            self.command_from_templates.append(h5["lang_description"][()])
            # Command being step-by-step plan generated in a scripted way.
            self.command_from_goals_templates.append(h5["lang_goal"][()])
            
            # Symbolic goals.
            sym_goal = h5["sym_goal"][()]
            sym_goal_values = h5["sym_values"][()]

            self.symbolic_goals.append(sym_goal)
            self.symbolic_goals_values.append(sym_goal_values)

            # Proces symbolic goals, depending on the settings.
            if cfg.process_goals == "clean":
                tokenized_goals = self.process_goals_clean(sym_goal, sym_goal_values, self.cfg.add_pad)
            elif cfg.process_goals == "sep":
                tokenized_goals = self.process_goals_sep(sym_goal, sym_goal_values, self.cfg.add_pad)
            else:
                raise ValueError(f"Invalid process_goal value '{cfg.process_goals}'")

            self.symbolic_goals_processed.append(" ".join(tokenized_goals))

            # Proces symbolic plans, depending on the settings.
            #print("Symbolic Plan / Actions: ", h5["sym_plan"][()], '\n')
            sym_plan = h5["sym_plan"][()]
            self.symbolic_plan.append(sym_plan)

            if cfg.process_plans == "clean":
                tokenized_plan, skipped_actions = self.process_plan_clean(sym_plan, self.cfg.skip_actions, self.cfg.add_pad)
            elif cfg.process_plans == "sep":
                tokenized_plan, skipped_actions = self.process_plan_sep(sym_plan, self.cfg.skip_actions, self.cfg.add_pad)
            elif cfg.process_plans == "split":
                tokenized_plan, skipped_actions = self.process_plan_split(sym_plan, self.cfg.skip_actions, self.cfg.add_pad)
            else:
                raise ValueError(f"Invalid process_goal value '{cfg.process_plans}'")

            self.symbolic_plan_processed.append(" ".join(tokenized_plan))
            self.symbolic_plan_skipped_actions.append(skipped_actions)

            # Set max, min, avg lengths.
            self.min_goals_length = min(self.min_goals_length, len(tokenized_goals))
            self.avg_goals_length += len(tokenized_goals)
            self.max_goals_length = max(self.max_goals_length, len(tokenized_goals))
            self.min_plan_length = min(self.min_plan_length, len(tokenized_plan))
            self.avg_plan_length += len(tokenized_plan)
            self.max_plan_length = max(self.max_plan_length, len(tokenized_plan))

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
        assert len(self.command_from_humans) == len(self.symbolic_goals)
        assert len(self.command_from_humans) == len(self.symbolic_goals_processed)
        assert len(self.command_from_humans) == len(self.symbolic_plan)
        assert len(self.command_from_humans) == len(self.symbolic_plan_processed)

        # Show basic split statistics.
        self.avg_goals_length = self.avg_goals_length / len(self.command_from_humans)
        self.avg_plan_length = self.avg_plan_length / len(self.command_from_humans)
        logging.info(f"Split '{self.cfg.split}' size = ", len(self.command_from_humans))
        logging.info(f"Number Goals: Min = {self.min_goals_length} Avg = {self.avg_goals_length} Max = {self.max_goals_length} ")
        logging.info(f"Plan length: Min = {self.min_plan_length} Avg = {self.avg_plan_length} Max = {self.max_plan_length} ")


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
    def process_goals_clean(cls, symbolic_goals, symbolic_goals_values, add_pad = True, return_string = False):
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

        if add_pad:
            tokenized_goals.extend(["[PAD]"])

        if return_string:
            return " ".join(tokenized_goals)
        else:
            return tokenized_goals

    @classmethod
    def process_goals_sep(cls, symbolic_goals, symbolic_goals_values, add_pad = True, return_string = False):
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

        if add_pad:
            tokenized_goals.extend(["[PAD]"])

        if return_string:
            return " ".join(tokenized_goals)
        else:
            return tokenized_goals

    @classmethod
    def process_plan_clean(cls, symbolic_plan, skip_actions, add_pad = True, return_string = False):
        """ Minimalistic plan processing, where all punctuation (brackets, commas) are kept. 

        Skips actions (DEFAULT):
        * APPROACH and GO_HOME actions are skipped!

        Returns:
            * list of str OR a single string (depending on the return_string flag)
            * list of skipped actions
        """
        # Split goals and goal values.
        symbolic_plan = symbolic_plan.split("),")

        tokenized_plans = []
        skipped_actions = []
        for i, action in enumerate(symbolic_plan):
            # Check if action should be skipped.
            skip = False
            for skip_action in skip_actions:
                if skip_action in action:
                    skipped_actions.append(i)
                    skip = True
                    break
            if skip:
                continue

            # Add removed "),"
            if action[-1] != ")":
                action = action + "),"
            # "Tokenize" action into items.
            action_items = action.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()

            tokenized_plans.extend(action_items)

        if add_pad:
            tokenized_plans.extend(["[PAD]"])

        if return_string:
            return " ".join(tokenized_plans), skipped_actions
        else:
            return tokenized_plans, skipped_actions

    @classmethod
    def process_plan_sep(cls, symbolic_plan, skip_actions, add_pad = True, return_string = False):
        """ A more sophisticated plan processing, where:
        * actions are separated by [SEP]
        * using [EOS] token after the last action
        * all punctuation (brackets, commas) are removed

        Skips actions (DEFAULT):
        * APPROACH and GO_HOME actions are skipped!

        Returns:
            * list of str OR a single string (depending on the return_string flag)
            * list of skipped actions
        """
        # Split goals and goal values.
        symbolic_plan = symbolic_plan.split("),")

        tokenized_plans = []
        skipped_actions = []
        for i, action in enumerate(symbolic_plan):
            # Check if action should be skipped.
            skip = False
            for skip_action in skip_actions:
                if skip_action in action:
                    skipped_actions.append(i)
                    skip = True
                    break
            if skip:
                continue

            # Add removed "),"
            if action[-1] != ")":
                action = action + " [SEP]"

            # "Tokenize" action into items.
            action_items = action.replace("(", " ").replace(")", " ").replace(",", " ").split()

            tokenized_plans.extend(action_items)

        # Add EOS + [PAD] at the end.
        tokenized_plans.extend(["[EOS]"])

        if add_pad:
            tokenized_plans.extend(["[PAD]"])

        if return_string:
            return " ".join(tokenized_plans), skipped_actions
        else:
            return tokenized_plans, skipped_actions


    @classmethod
    def process_plan_split(cls, symbolic_plan, skip_actions, add_pad = True, return_string = False):
        """ A complex plan processing:
        * Returns a standardized format: verb - main object - supporting object (OPTIONAL) - [SEP/EOS] token
        * verbs are split into verb + supporting object
        * all punctuation (brackets, commas) are removed
        * actions are separated by [SEP]
        * [EOS] token after the last action

        Skips actions (DEFAULT):
        * APPROACH and GO_HOME actions are skipped!

        Returns:
            * list of str OR a single string (depending on the return_string flag)
            * list of skipped actions
        """
        # Split goals and goal values.
        #print(symbolic_plan)
        symbolic_plan = symbolic_plan.split("),")

        tokenized_plans = [] # List of lists.
        skipped_actions = []
        for i, action in enumerate(symbolic_plan):
            # Check if action should be skipped.
            skip = False
            for skip_action in skip_actions:
                if skip_action in action:
                    skipped_actions.append(i)
                    skip = True
                    break
            if skip:
                continue

            # Add separator.
            if action[-1] != ")":
                action = action + " [SEP]"

            # "Tokenize" plan into actions.
            action_items = action.replace("(", " ").replace(")", " ").replace(",", " ").split()

            verb = action_items[0]
            main_object = action_items[-2] # SEP is -1.
            supporting_object = None
            end_token = action_items[-1]
            
            # Try to find "from" and "on" indices.
            _from_idx = verb.find('_from_')
            _on_idx = verb.find('_on_')

            # Extract the supporting objects and truncate verbs accordingly.
            if _from_idx >= 0:
                supporting_object = verb[(_from_idx + 6):]
                verb = verb[:_from_idx]

            elif _on_idx >= 0:
                supporting_object = verb[(_on_idx + 4):]
                verb = verb[:_on_idx]

            elif verb.startswith("stack"):
                _idx = verb.find("_on")
                # Flip objects!
                supporting_object = main_object
                main_object = verb[6:_idx]
                verb = "stack"

            elif verb.startswith("align"):
                _idx = verb.find("_with")
                # Flip objects!
                supporting_object = main_object
                main_object = verb[6:_idx]
                verb = 'align'

            # Clearn verbs a bit.
            verb=verb.replace("_obj", "")

            if supporting_object is None:
                action_items = [verb, main_object, end_token]
            else:
                action_items = [verb, main_object, supporting_object, end_token]
                # Check if supporting object was present in previous action.
                if verb == "lift" and len(tokenized_plans[-1]) == 3:
                    # Add the supporting object to the previous (grasp) action.
                    tokenized_plans[-1].insert(2, supporting_object)
            #print(action_items)

            # Finally add everything to plan.
            tokenized_plans.append(action_items)

        #print(tokenized_plans)
        #import pdb;pdb.set_trace()
        # Flatten the plan.
        tokenized_plans = [token for action in tokenized_plans for token in action]
        # Replace last token with [EOS]
        tokenized_plans[-1] = "[EOS]"

        # Add additional [PAD] token if required.
        if add_pad:
            tokenized_plans.extend(["[PAD]"])

        # Return processed plan.
        if return_string:
            return " ".join(tokenized_plans), skipped_actions
        else:
            return tokenized_plans, skipped_actions

    def __getitem__(self, idx):

        all_commands = []
        # Create list of possible commands.
        command_from_humans = self.command_from_humans[idx]
        if type(command_from_humans) == str:
            all_commands.append(command_from_humans)
        else:
            all_commands.extend(command_from_humans)
            # Randomly pick one of the command humans to be returned.
            command_from_humans = command_from_humans[np.random.randint(len(command_from_humans))]

        # Add template commands.
        all_commands.append(self.command_from_templates[idx])
        # Add template commands from goals.
        all_commands.append(self.command_from_goals_templates[idx])
        
        # Sample command randomly from one of three sources.
        command = all_commands[np.random.randint(len(all_commands))]

        # Create sample.
        sample = {
            "idx": idx,
            "filename": self.filename[idx],
            "command": command,
            "command_from_humans": command_from_humans,
            "command_from_templates": self.command_from_templates[idx],
            "command_from_goals_templates": self.command_from_goals_templates[idx],
            "symbolic_goals": self.symbolic_goals[idx],
            "symbolic_goals_processed": self.symbolic_goals_processed[idx],
            "symbolic_plan": self.symbolic_plan[idx],
            "symbolic_plan_processed": self.symbolic_plan_processed[idx],
        }

        #print(sample)
        # Add
        if self.cfg.return_rgb:
            sample["init_rgb"] = self.init_rgb_images[idx]

        return sample


if __name__ == "__main__":
    # Create dataset.
    sierra_cfg = SierraDatasetConf(brain_path="/home/tkornuta/data/brain2", return_rgb=False, limit=10)
    sierra_ds = SierraDataset(cfg=sierra_cfg)

    # Create dataloader and get batch.
    sierra_dl = DataLoader(sierra_ds, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(sierra_dl))

    # Show samples.
    #print(batch.keys())    
    for i in range(len(batch["idx"])):
        print("="*100)
        for k,v in batch.items():
            if k == "init_rgb":
                continue
            print(f"{k}: {v[i]}\n")
