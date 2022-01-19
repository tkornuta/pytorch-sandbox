# SPDX-License-Identifier: Apache-2.0

import os
from typing import List
import h5py
import csv
from tqdm import tqdm

brain_path = "/home/tkornuta/data/brain2"
sierra_path = os.path.join(brain_path, "leonardo_sierra")
output_path = os.path.join(brain_path, "processed")

# Get files.
sierra_files = [f for f in os.listdir(sierra_path) if os.path.isfile(os.path.join(sierra_path, f))]

# Open csv file with commands created by humans.
command_humans_dict = {}
with open(os.path.join(brain_path, 'sierra_5k_v1.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        command_humans_dict[row[0][:-4]] = row[1:]

# Prepare data structures
sample_names = []
command_templates = []
command_humans = []
symbolic_plans = []
symbolic_goals = []

# Open files one by one.
for filename in tqdm(sierra_files):
    # Load the file.
    h5 = h5py.File(os.path.join(sierra_path, filename), 'r')
    
    # Add sample name.
    sample_id = filename[:-3]
    sample_names.append(sample_id)

    # Short command generated in a scripted way.
    #print("Command (templated): ", h5["lang_description"][()], '\n')
    command_templates.append(h5["lang_description"][()])
    # A step-by-step plan generated in a scripted way.
    #print("Plan language: ", h5["lang_plan"][()], '\n')
    #print(command_templates[-1])

    # Human command - from another file!
    command_humans.append(command_humans_dict[sample_id])

    #print("Symbolic Plan / Actions: ", h5["sym_plan"][()], '\n')
    symbolic_plans.append(h5["sym_plan"][()])

    #print("Symbolic Goal: ", h5["sym_goal"][()], '\n')
    symbolic_goals.append(h5["sym_goal"][()])

    #import pdb;pdb.set_trace()

# Save to files.
os.makedirs(output_path, exist_ok=True)

def save_to(name, list_to_save):
    filename = os.path.join(output_path, name)
    with open(filename, "w") as f:
        for obj in list_to_save:
            if type(obj) is list:
                for item in obj:
                    f.write(item + ';')
                f.write('\n')
            else:
                f.write(obj + ';\n')

    print(f"List saved to `{filename}`")

save_to("sample_names.csv", sample_names)
save_to("command_templates.csv", command_templates)
save_to("command_humans.csv", command_humans)
save_to("symbolic_plans.csv", symbolic_plans)
save_to("symbolic_goals.csv", symbolic_goals)
