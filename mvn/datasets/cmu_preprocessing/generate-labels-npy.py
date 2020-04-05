"""
    Generate 'labels.npy' for multiview 'human36m.py'
    from https://github.sec.samsung.net/RRU8-VIOLET/multi-view-net/

    Usage: `python3 generate-labels-npy.py <path/to/cmu-panoptic-data-root>`
"""

# TODO: Modify this to fit our needs

import os, sys
import numpy as np
import h5py
import re

# Change this line if you want to use Mask-RCNN or SSD bounding boxes instead of H36M's "ground truth".
BBOXES_SOURCE = 'GT' # or 'MRCNN' or 'SSD'

retval = {
    'subject_names': [],
    'camera_names': [],
    'scene_names': []
}

cmu_root = sys.argv[1]

destination_file_path = os.path.join(
    cmu_root, f"cmu-multiview-labels-{BBOXES_SOURCE}bboxes.npy")

'''
FORMATTING/ORGANISATION OF FOLDERS & FILES

Images:
    $DIR_ROOT/[SCENE_NAME]/hdImgs/[VIEW_ID]/[VIEW_ID]_[FRAME_ID].jpg
    (e.g.) ./171026_pose1/hdImgs/00_24/00_24_00012200.jpg

Pose Data:
    $DIR_ROOT/[SCENE_NAME]/hdPose3d_stage1_coco19/body3DScene_[FRAME_ID].jpg
    (e.g.) ./171026_pose1/hdPose3d_stage1_coco19/body3DScene_00012200.json

    JSON data has this notable format:
    "bodies":[
        {
            "id": [PERSON_ID],
            "joints": [ ARRAY OF JOINT COORDINATES IN COCO 19 FORMAT]
        },
        {
            ...
        }
    ] 

Camera Calibration Data:
    $DIR_ROOT/[SCENE_NAME]/calibration_[SCENE_NAME].json
'''

'''
Search through file system for 
1) scene names
    a) available cameras
    b) available frames
    c) people
'''
# Loop thru directory files and find scene names
for scene_name in os.listdir(cmu_root):
    # Make sure that this is actually a scene
    if "_pose" not in scene_name:
        continue

    print(scene_name)

# Generate cameras based on len of names
retval['cameras'] = np.empty(
    (len(retval['subject_names']), len(retval['camera_names'])),
    dtype=[
        ('R', np.float32, (3,3)),
        ('t', np.float32, (3,1)),
        ('K', np.float32, (3,3)),
        ('dist', np.float32, 5)
    ]
)

table_dtype = np.dtype([
    ('subject_idx', np.int8),
    ('action_idx', np.int8),
    ('frame_idx', np.int16),
    ('keypoints', np.float32, (17,3)), # roughly MPII format
    ('bbox_by_camera_tlbr', np.int16, (len(retval['camera_names']),4))
])
retval['table'] = []


# TODO: COPY BACK FROM HUMAN36M PREPROCESSING FILE

retval['table'] = np.concatenate(retval['table'])
assert retval['table'].ndim == 1

print("Total frames in Human3.6Million:", len(retval['table']))
np.save(destination_file_path, retval)
