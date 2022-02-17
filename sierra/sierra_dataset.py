# SPDX-License-Identifier: Apache-2.0

import os
import csv
import io
from typing import List
import numpy as np
from PIL import Image

from dataclasses import dataclass, field

import h5py
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from brain_gym import logger


@dataclass
class SierraDatasetConf:
    type: str = "brain_gym.data_modules.sierra_dataset_clean.SierraDataset"
    brain_path: str = "/home/tkornuta/data/brain2"
    data_subpath: str = "leonardo_sierra"
    command_sources: List = field(default_factory=lambda: ["humans", "lang_description", "lang_goal"]) # List of sources of commands.
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

        # Transforms used to reshape/normalize RGB images to format/standard used in pretrained ResNet/ViT models.
        rgb_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([224, 224]),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # Statistics.
        self.min_command_words = 100000
        self.avg_command_words = 0
        self.max_command_words = 0

        self.min_goals_length = 100000
        self.avg_goals_length = 0
        self.max_goals_length = 0

        self.min_plan_length = 100000
        self.avg_plan_length = 0
        self.max_plan_length = 0

        # Get files - adequate to the split.
        if cfg.split == "all":
            # Get all files.
            sierra_files = [f for f in os.listdir(sierra_path) if os.path.isfile(os.path.join(sierra_path, f))]
        elif cfg.split in ["train", "valid", "test"]:
            split_file = os.path.join(cfg.brain_path, cfg.split + ".csv")
            if not os.path.exists(split_file):
                # Create all splits.
                self.split_h5_by_code()
            # Load list of files.
            sierra_files = self.get_files(split_file)
        else:
            raise ValueError("Invalid ")
        
        # List containing all samples.
        self.samples =[]
        num_records_processed = 0
        # Open files one by one.
        for i,filename in enumerate(tqdm(sierra_files)):

            # Limit number of samples - for testing purposes.
            if cfg.limit >0 and len(self.samples) >= cfg.limit:
                break

            # Load the file.
            h5 = h5py.File(os.path.join(sierra_path, filename), 'r')

            # Add sample name.
            sample_id = filename[:-3]

            # Create new sample and store there all the fields.
            sample = {}
            sample["filename"] = sample_id

            # Symbolic goals.
            sym_goal = h5["sym_goal"][()]
            sym_goal_values = h5["sym_values"][()]
            sample["symbolic_goals"] = sym_goal
            # Change to strings - to avoiding changing to tensors.
            sample["symbolic_goals_values"] = ",".join([str(v) for v in sym_goal_values])

            # Proces symbolic goals, depending on the settings.
            if cfg.process_goals == "clean":
                tokenized_goals = self.process_goals_clean(sym_goal, sym_goal_values, self.cfg.add_pad)
            elif cfg.process_goals == "sep":
                tokenized_goals = self.process_goals_sep(sym_goal, sym_goal_values, self.cfg.add_pad)
            else:
                raise ValueError(f"Invalid process_goal value '{cfg.process_goals}'")

            sample["symbolic_goals_processed"] = " ".join(tokenized_goals)

            # Proces symbolic plans, depending on the settings.
            sym_plan = h5["sym_plan"][()]
            sample["symbolic_plan"] = sym_plan

            if cfg.process_plans == "clean":
                tokenized_plan, skipped_actions = self.process_plan_clean(sym_plan, self.cfg.skip_actions, self.cfg.add_pad)
            elif cfg.process_plans == "sep":
                tokenized_plan, skipped_actions = self.process_plan_sep(sym_plan, self.cfg.skip_actions, self.cfg.add_pad)
            elif cfg.process_plans == "split":
                tokenized_plan, skipped_actions = self.process_plan_split(sym_plan, self.cfg.skip_actions, self.cfg.add_pad)
            else:
                raise ValueError(f"Invalid process_goal value '{cfg.process_plans}'")

            sample["symbolic_plan_processed"] = " ".join(tokenized_plan)

            # Change to strings - to avoiding changing to tensors.
            sample["symbolic_plan_skipped_actions"] = ",".join([str(v) for v in skipped_actions])

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
                
                sample["init_rgb"] = rgb_normalized_tensor

            # Finally, create list of commands that are available for this sample.
            commands = []

            # Process "humans" if indicated.
            if "humans" in self.cfg.command_sources:
                human_commands = command_from_humans_dict[sample_id]
                # Check if sample is VALID!
                if type(human_commands) == list:
                    if len(human_commands) == 0:
                        # Error for sample 4759 command: []
                        logger.warning(f"Skipping humand commands for {i}-th sample `{sample_id}`")
                    else:                    
                        # Several possible commands.
                        commands.extend(human_commands)
                else:
                    # Single command.
                    commands.append(human_commands)

            # Process "lang_description" if indicated - step-by-step plan generated in a scripted way
            if "lang_description" in self.cfg.command_sources:
                lang_descriptions = h5["lang_description"][()]
                commands.append(lang_descriptions)

            # Process "lang_goal" if indicated - command generated in a scripted way.
            if "lang_goal" in self.cfg.command_sources:
                lang_goal = h5["lang_goal"][()]
                
                commands.extend(lang_goal.split(","))

            #print(f"Resulting commands ({len(commands)}) = {commands}")

            # Ok, now make n-copies the same sample for each command.
            for command in commands:
                # Copy sample.
                sample_copy = {
                    "idx": len(self.samples),
                    "command": command,
                }
                for k,v in sample.items():
                    sample_copy[k] = v
                self.samples.append(sample_copy)

                # Update statistics.
                len_command = len(command.split())
                len_goals = len(tokenized_goals)
                len_plan = len(tokenized_plan)

                self.min_command_words = min(self.min_command_words, len_command)
                self.avg_command_words += len_command
                self.max_command_words = max(self.max_command_words, len_command)
                self.min_goals_length = min(self.min_goals_length, len_goals)
                self.avg_goals_length += len_goals
                self.max_goals_length = max(self.max_goals_length, len_goals)
                self.min_plan_length = min(self.min_plan_length, len_plan)
                self.avg_plan_length += len_plan
                self.max_plan_length = max(self.max_plan_length, len_plan)

                # Stop adding samples if limit set.
                if cfg.limit > 0 and len(self.samples) >= cfg.limit:
                    break
            
            # Increment number of processed samples.
            num_records_processed += 1

        # Show basic split statistics.
        self.avg_command_words = int(self.avg_command_words / len(self.samples))
        self.avg_goals_length = int(self.avg_goals_length / len(self.samples))
        self.avg_plan_length = int(self.avg_plan_length / len(self.samples))
        logger.info(f"Split '{self.cfg.split}' size = {len(self.samples)}")
        logger.info(f"Number of records processed = {num_records_processed}")
        logger.info(f"Number of words in commands | Min = {self.min_command_words} | Avg = {self.avg_command_words} | Max = {self.max_command_words}")
        logger.info(f"Number of token goals | Min = {self.min_goals_length} | Avg = {self.avg_goals_length} | Max = {self.max_goals_length}")
        logger.info(f"Number of tokens in plan | Min = {self.min_plan_length} | Avg = {self.avg_plan_length} | Max = {self.max_plan_length}")


    def split_h5_by_code(self):
        """ Split directory by code. Create a bunch of different files containing
        lists of data files used for train/val/test.

        Held-out "test" examples are determined by task code. """

        logger.info("Regenerating data splits...")

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
                logger.error('Problem handling file: ' + str(filename))
                logger.error('Full filename: ' + str(full_filename))
                logger.error('Failed with exception: ' + str(e))
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
            logger.warning("Split finished with errors. Had to skip the following files: " + str(skipped))
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
        return len(self.samples)

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

        # Add [EOS] at the end.
        if tokenized_plans[-1] == "[SEP]":
            tokenized_plans[-1] = ["[EOS]"]
        else:
            tokenized_plans.extend(["[EOS]"])

        # Optionally add [PAD] at the end.
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

            # Clean verbs a bit.
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
        # Just return the sample :)
        return self.samples[idx]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    # Create dataset.
    sierra_cfg = SierraDatasetConf(brain_path="/home/tkornuta/data/brain2", return_rgb=False, split="test")#, command_sources=["humans"])#limit=10)
    sierra_ds = SierraDataset(cfg=sierra_cfg)

    # Create dataloader and get batch.
    sierra_dl = DataLoader(sierra_ds, batch_size=1, shuffle=True, num_workers=0)
    batch = next(iter(sierra_dl))

    # Show samples.
    #print(batch.keys())    
    for i in range(len(batch["idx"])):
        print("="*100)
        for k,v in batch.items():
            if k == "init_rgb":
                continue
            print(f"{k}: {v[i]}")
