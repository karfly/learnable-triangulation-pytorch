#!/bin/python

"""
    Generate 'labels.npy' for multiview 'human36m.py'
    from https://github.sec.samsung.net/RRU8-VIOLET/multi-view-net/

    Usage: `python3 generate-labels-npy.py <path/to/cmu-panoptic-data-root>`
"""

# TODO: Modify this to fit our needs

import os, sys
import numpy as np
import json

def jsonToDict(filename):
    # Read file
    with open(filename, 'r') as f:
        data = f.read()

    # parse file
    return json.loads(data)


# Change this line if you want to use Mask-RCNN or SSD bounding boxes instead of H36M's "ground truth".
BBOXES_SOURCE = 'GT' # or 'MRCNN' or 'SSD'

retval = {
    'subject_names': [],
    'camera_names': [],
    'pose_names': []
}

cmu_root = sys.argv[1]

destination_file_path = os.path.join(
    cmu_root, "extra", f"cmu-multiview-labels-{BBOXES_SOURCE}bboxes.npy")

'''
FORMATTING/ORGANISATION OF FOLDERS & FILES

Images:
    $DIR_ROOT/[POSE_NAME]/hdImgs/[VIEW_ID]/[VIEW_ID]_[FRAME_ID].jpg
    (e.g.) ./171026_pose1/hdImgs/00_24/00_24_00012200.jpg

Pose Data:
    $DIR_ROOT/[POSE_NAME]/hdPose3d_stage1_coco19/body3DScene_[FRAME_ID].jpg
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
    $DIR_ROOT/[POSE_NAME]/calibration_[POSE_NAME].json
'''

'''
Search through file system for 
1) pose names
    a) available cameras
    b) available frames
    c) people
'''
# Parse camera data and return a dictionary
# of better formatted data, with 
# key: camera name, value: dictionary of intrinsics
def parseCameraData(filename):
    info_array = jsonToDict(filename)["cameras"]
    data = {}

    for camera_params in info_array:
        name = camera_params["name"]
        data[name] = {}

        data[name]['R'] = np.array(camera_params['R'])
        data[name]['t'] = np.array(camera_params['t'])
        data[name]['K'] = np.array(camera_params['K'])
        data[name]['dist'] = np.array(camera_params['distCoef'])

    return data

def parseJointsData(joints_data):
    # TODO: Parse somehow: Numpy array?
    return np.array(joints_data)

def parsePersonData(filename):
    info_array = jsonToDict(filename)["bodies"]
    people_array = []

    for person_data in info_array:
        joints = parseJointsData(person_data["joints19"])
        people_array.append(joints)
    
    return people_array

# Loop thru directory files and find scene names
for pose_name in os.listdir(cmu_root):
    # Make sure that this is actually a scene
    # and not sth like "scripts" or "matlab"
    if "_pose" not in pose_name:
        continue

    retval["pose_names"].append(pose_name)

    pose_dir = os.path.join(cmu_root, pose_name)

    # Retrieve camera calibration data
    calibration_file = os.path.join(pose_dir, f"calibration_{pose_name}.json")
    camera_data = parseCameraData(calibration_file)

    # Count the frames by adding them to the dictionary
    # Only the frames with correct length are valid
    # Otherwise have missing data/images --> ignore
    frame_cnt = {}
    camera_names = []
    person_data_path = os.path.join(pose_dir, "hdPose3d_stage1_coco19")

    for frame_name in os.listdir(person_data_path):
        person_data_filename = os.path.join(person_data_path, frame_name);
        person_data = parsePersonData(person_data_filename)
        #print(frame_name, end=" "); print(person_data)

        frame_name = frame_name.replace("body3DScene_","").replace(".json","")
        frame_cnt[frame_name] = 1

    # Find the cameras
    images_dir = os.path.join(pose_dir, "hdImgs")
    for camera_name in os.listdir(images_dir):
        # Populate frames dictionary
        images_dir_cam = os.path.join(images_dir, camera_name)

        for frame_name in os.listdir(images_dir_cam):
            frame_name = frame_name.replace(f"{camera_name}_","").replace(".jpg","").replace(".png","")

            if frame_name in frame_cnt:
                frame_cnt[frame_name] += 1

        camera_names.append(camera_name)

    # Only frames with full count are counted
    valid_frames = []
    for frame_name in frame_cnt:
        if frame_cnt[frame_name] == 1 + len(camera_names):
            valid_frames.append(frame_name)

    del frame_cnt
    print(pose_name, end=" "); print(valid_frames)

    retval["camera_names"] += camera_names

retval["camera_names"] = list(set(retval["camera_names"]))

print(retval)
exit()

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
    ('pose_idx', np.int8),
    ('frame_idx', np.int16),
    ('keypoints', np.float32, (17,3)), # roughly MPII format
    ('bbox_by_camera_tlbr', np.int16, (len(retval['camera_names']),4))
])
retval['table'] = []


# TODO: COPY BACK FROM HUMAN36M PREPROCESSING FILE

retval['table'] = np.concatenate(retval['table'])
assert retval['table'].ndim == 1

print("Total frames in CMU Panoptic Dataset:", len(retval['table']))
np.save(destination_file_path, retval)
