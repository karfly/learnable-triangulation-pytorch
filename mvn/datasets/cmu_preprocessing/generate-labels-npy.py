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
    'camera_names': set(),
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
        # make it a number
        name = camera_params["name"].replace("_", "")

        data[name] = {}

        data[name]['R'] = np.array(camera_params['R'])
        data[name]['t'] = np.array(camera_params['t'])
        data[name]['K'] = np.array(camera_params['K'])
        data[name]['dist'] = np.array(camera_params['distCoef'])

    return data

def parseJointsData(joints_data):
    # TODO: Number of joints seem to be in the form of (19,4)
    # TODO: Homogeneous coordinates?
    return np.array(joints_data)

def parsePersonData(filename):
    info_array = jsonToDict(filename)["bodies"]
    people_array = []

    for person_data in info_array:
        joints = parseJointsData(person_data["joints19"])
        people_array.append(joints)
    
    return people_array

# In CMU data everything is by pose, not by person
# NOTE: Calibration data for CMU is different for every pose, although only slightly :(
data_by_pose = {}

# Loop thru directory files and find scene names
for pose_name in os.listdir(cmu_root):
    # Make sure that this is actually a scene
    # and not sth like "scripts" or "matlab"
    if "_pose" not in pose_name:
        continue

    data = {}

    retval["pose_names"].append(pose_name)

    pose_dir = os.path.join(cmu_root, pose_name)
    data["pose_dir"] = pose_dir

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

        camera_name = camera_name.replace("_", "")

        retval["camera_names"].add(camera_name)
        camera_names.append(camera_name)

    # Only frames with full count are counted
    valid_frames = []
    person_data = {} # by frame name

    for frame_name in frame_cnt:
        if frame_cnt[frame_name] == 1 + len(camera_names):
            valid_frames.append(frame_name)
            
            person_data_filename = os.path.join(person_data_path, frame_name)
            person_data_arr = parsePersonData(person_data_filename)

            person_data[frame_name] = person_data_arr

    del frame_cnt

    data["valid_frames"] = valid_frames
    data["person_data"] = person_data
    data["camera_names"] = camera_names
    data["camera_names"].sort()

    # Generate camera data
    data["camera_data"] = {}
    for camera_name in data["camera_names"]:
        data["camera_data"][camera_name] = camera_data[camera_name]

    data_by_pose[pose_name] = data

print(data_by_pose)

retval['camera_names'] = list(retval['camera_names'])

# Generate cameras based on len of names
# Note that camera calibrations are different for each pose
retval['cameras'] = np.empty(
    (len(retval['pose_names']), len(retval['camera_names'])),
    dtype=[
        ('R', np.float32, (3, 3)),
        ('t', np.float32, (3, 1)),
        ('K', np.float32, (3, 3)),
        ('dist', np.float32, 5)
    ]
)

# Now that we have collated the data into easier-to-parse ways
# Need to reorganise data into return values needed for dataset class 

# Each pose, person has different entry
table_dtype = np.dtype([
    ('pose_name', np.int8), 
    ('subject_idx', np.int8),
    ('frame_names', np.int16),
    ('keypoints', np.float32, (19, 4)),  # roughly MPII format
    ('bbox_by_camera_tlbr', np.int16, (len(retval['camera_names']), 4))
])

retval['table'] = []

for pose_idx, pose_name in enumerate(data_by_pose):
    data = data_by_pose[pose_name]

    for person_idx, person_name in enumerate(data[""]):
        table_segment = np.empty(len(data["valid_frames"]), dtype=table_dtype)

        # TODO: Poses changing from CMU to H36M, if the current one doesn't do it automatically

        table_segment['person_idx'] = person_idx
        table_segment['pose_names'] = pose_name
        table_segment['frame_names'] = data["valid_frames"]
        table_segment['keypoints'] = poses_world

        # let a (0,0,0,0) bbox mean that this view is missing
        table_segment['bbox_by_camera_tlbr'] = 0

        retval['table'].append(table_segment)

exit()

# TODO: COPY BACK FROM HUMAN36M PREPROCESSING FILE

retval['table'] = np.concatenate(retval['table'])
assert retval['table'].ndim == 1

print("Total frames in CMU Panoptic Dataset:", len(retval['table']))
np.save(destination_file_path, retval)
