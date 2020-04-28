#!/bin/python

'''
Collate BBOX data from CMU Panoptic Dataset into a npy file
'''

USAGE_PROMPT="""
$ python3 collect-bboxes-npy.py <path/to/mrcnn-detections/folder> <path/to/output/file> <1-for-debug(optional)>

Example:
$ python3 collect-bboxes-npy.py $THIS_REPOSITORY/data/pretrained/cmu/mrcnn-detections $THIS_REPOSITORY/data/pretrained/cmu
"""

import os, sys
import numpy as np
import json
from collections import defaultdict

DEBUG = False

def jsonToDict(filename):
    # Read file
    with open(filename, 'r') as f:
        data = f.read()

    # parse file
    return json.loads(data)

try:
    bbox_dir = sys.argv[1]
    output_dir = sys.argv[2]
except:
    print(USAGE_PROMPT)
    exit()

try:
    DEBUG = bool(sys.argv[3])
except:
    DEBUG = False

print(f"Debug mode: {DEBUG}\n")

destination_file_path = os.path.join(output_dir, "cmu-bboxes.npy")

# BBOX Data
def nesteddict(): return defaultdict(nesteddict)

bbox_data = nesteddict()

#bbox_dir = os.path.join(bbox_root, 'mrcnn-detections')

print("Collecting BBOXes...")
assert os.path.isdir(bbox_dir), "Invalid BBOX directory '%s'\n%s" % (bbox_dir, USAGE_PROMPT)

for action_name in os.listdir(bbox_dir):
    # Make sure that this is actually a scene
    # and not sth like 'scripts' or 'matlab'
    if '_pose' not in action_name:
        continue

    bbox_action_dir = os.path.join(bbox_dir, action_name, 'mrcnn-detections')

    if not os.path.isdir(bbox_action_dir):
        continue

    bbox_data[action_name] = {}

    for camera_name in os.listdir(bbox_action_dir):
        bbox_data_arr = jsonToDict(os.path.join(bbox_action_dir, camera_name))

        camera_name = camera_name.replace(".json", "")

        bbox_data[action_name][camera_name] = np.array(bbox_data_arr)

        if DEBUG:
            print(action_name, camera_name)

print("Done!\nSaving bbox npy file...")

def freeze_defaultdict(x):
    x.default_factory = None
    for value in x.values():
        if type(value) is defaultdict:
            freeze_defaultdict(value)

# convert to normal dict
try:
    freeze_defaultdict(bbox_data)

    np.save(destination_file_path, bbox_data)
    print(f"BBOX npy file saved to {destination_file_path}")
except:
    raise f"Failed to save file {destination_file_path}"
