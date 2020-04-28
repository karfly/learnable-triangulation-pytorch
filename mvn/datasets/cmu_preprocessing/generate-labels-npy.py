#!/bin/python

'''
    Generate `labels.npy` for multiview `cmupanoptic.py`
    Usage: `python3 generate-labels-npy.py <path/to/cmu-panoptic-data-root> <path/to/cmu-mrcnn-bbox-detections>`
'''

# TODO: Modify this to fit our needs

import os, sys
import numpy as np
import json
import pickle

USAGE_PROMPT = """
$ python3 generate-lables-npy.py <path/to/data> <path/to/bbox-npy-file>

Example (default):
$ python3 generate-lables-npy.py $THIS_REPOSITORY/data/cmupanoptic $THIS_REPOSITORY/data/cmupanoptic/cmu-bboxes.npy
"""

def jsonToDict(filename):
    # Read file
    with open(filename, 'r') as f:
        data = f.read()

    # parse file
    return json.loads(data)


# Change this line if you want to use Mask-RCNN or SSD bounding boxes instead of H36M's 'ground truth'.
BBOXES_SOURCE = 'MRCNN' # or 'MRCNN' or 'SSD'
DEBUG = False

retval = {
    'camera_names': set(),
    'action_names': []
}

try:
    cmu_root = sys.argv[1]
    bbox_root = sys.argv[2]
except:
    print("Usage: ",USAGE_PROMPT)
    exit()

destination_file_path = os.path.join(
    cmu_root, f'cmu-multiview-labels-{BBOXES_SOURCE}bboxes.npy')

assert os.path.isdir(cmu_root), "Invalid data directory '%s'\n%s" % (cmu_root, USAGE_PROMPT)

'''
FORMATTING/ORGANISATION OF FOLDERS & FILES

Images:
    $DIR_ROOT/[ACTION_NAME]/hdImgs/[VIEW_ID]/[VIEW_ID]_[FRAME_ID].jpg
    (e.g.) ./171026_pose1/hdImgs/00_24/00_24_00012200.jpg

Pose Data:
    $DIR_ROOT/[ACTION_NAME]/hdPose3d_stage1_coco19/body3DScene_[FRAME_ID].jpg
    (e.g.) ./171026_pose1/hdPose3d_stage1_coco19/body3DScene_00012200.json

    JSON data has this notable format:
    'bodies':[
        {
            'id': [PERSON_ID],
            'joints': [ ARRAY OF JOINT COORDINATES IN COCO 19 FORMAT]
        },
        {
            ...
        }
    ] 

Camera Calibration Data:
    $DIR_ROOT/[ACTION_NAME]/calibration_[ACTION_NAME].json
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
    info_array = jsonToDict(filename)['cameras']
    data = {}

    for camera_params in info_array:
        # make it a number
        name = camera_params['name']

        data[name] = {}

        data[name]['R'] = camera_params['R']
        data[name]['t'] = camera_params['t']
        data[name]['K'] = camera_params['K']
        data[name]['dist'] = camera_params['distCoef']

    return data

# Number of joints are in the form of (19,4)
# Return an np array accordingly
def parseJointsData(joints_data):
    return np.array(joints_data).reshape((19,4))

def parsePersonData(filename):
    info_array = jsonToDict(filename)['bodies']
    people_array = []

    for person_data in info_array:
        D = {}
        D['joints'] = parseJointsData(person_data['joints19'])
        D['id'] = int(person_data['id']) # note: may have only 1 body, but id = 4

        people_array.append(D)
    
    return people_array


def square_the_bbox(bbox):
    top, left, bottom, right = bbox
    width = right - left
    height = bottom - top

    if height < width:
        center = (top + bottom) * 0.5
        top = int(round(center - width * 0.5))
        bottom = top + width
    else:
        center = (left + right) * 0.5
        left = int(round(center - height * 0.5))
        right = left + height

    return top, left, bottom, right

def parseBBOXData(bbox_dir):
    bboxes = np.load(bbox_dir, allow_pickle=True).item()

    return bboxes

    for action in bboxes.keys():
        for camera, bbox_array in bboxes[action].items():
            for frame_idx, bbox in enumerate(bbox_array):
                bbox[:] = square_the_bbox(bbox)

    return bboxes

if BBOXES_SOURCE == 'MRCNN':
    bbox_data = parseBBOXData(bbox_root)
    
    print(f"{BBOXES_SOURCE} bboxes loaded!\n")
else:
    # NOTE: If you are not using the provided MRCNN detections, you have to implement the parser yourself
    raise NotImplementedError

# In CMU data everything is by pose, not by person/subject
# NOTE: Calibration data for CMU is different for every pose, although only slightly :(
data_by_pose = {}

print("Generating labels...")
# Loop thru directory files and find scene names
for action_name in os.listdir(cmu_root):
    # Make sure that this is actually a scene
    # and not sth like 'scripts' or 'matlab'
    if '_pose' not in action_name:
        continue

    data = {}

    action_dir = os.path.join(cmu_root, action_name)
    
    # Ensure is a proper directory
    if not os.path.isdir(action_dir):
        continue

    retval['action_names'].append(action_name)
    data['action_dir'] = action_dir

    # Retrieve camera calibration data
    calibration_file = os.path.join(action_dir, f'calibration_{action_name}.json')
    camera_data = parseCameraData(calibration_file)

    # Count the frames by adding them to the dictionary
    # Only the frames with correct length are valid
    # Otherwise have missing data/images --> ignore
    frame_cnt = {}
    camera_names = []
    person_data_path = os.path.join(action_dir, 'hdPose3d_stage1_coco19')

    for frame_name in os.listdir(person_data_path):
        frame_name = frame_name.replace('body3DScene_','').replace('.json','')
        frame_cnt[frame_name] = 1

    # Find the cameras
    images_dir = os.path.join(action_dir, 'hdImgs')
    for camera_name in os.listdir(images_dir):
        # Populate frames dictionary
        images_dir_cam = os.path.join(images_dir, camera_name)

        for frame_name in os.listdir(images_dir_cam):
            frame_name = frame_name.replace(f'{camera_name}_','').replace('.jpg','').replace('.png','')

            if frame_name in frame_cnt:
                frame_cnt[frame_name] += 1

        retval['camera_names'].add(camera_name)
        camera_names.append(camera_name)

    # Only frames with full count are counted
    valid_frames = []
    person_data = {} # by frame name

    for frame_name in frame_cnt:
        if frame_cnt[frame_name] == 1 + len(camera_names):
            valid_frames.append(frame_name)
            
            person_data_filename = os.path.join(person_data_path, f'body3DScene_{frame_name}.json')
            person_data_arr = parsePersonData(person_data_filename)

            person_data[frame_name] = person_data_arr

    del frame_cnt

    data['valid_frames'] = sorted(valid_frames)
    data['person_data'] = person_data
    data['camera_names'] = sorted(camera_names)

    # Generate camera data
    data['camera_data'] = {}
    for camera_name in data['camera_names']:
        data['camera_data'][camera_name] = camera_data[camera_name]

    data_by_pose[action_name] = data

# Consolidate camera names and sort them
retval['camera_names'] = list(retval['camera_names'])
retval['camera_names'].sort()

# Generate cameras based on len of names
# Note that camera calibrations are different for each pose
retval['cameras'] = np.empty(
    (len(retval['action_names']), len(retval['camera_names'])),
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
    ('action_idx', np.int8), 
    ('person_id', np.int8),
    ('frame_names', np.int16),
    ('keypoints', np.float32, (19, 4)),  # roughly MPII format
    ('bbox_by_camera_tlbr', np.int16, (len(retval['camera_names']), 4))
])

retval['table'] = []

# Iterate through the poses to fill up the table and camera data
for action_idx, action_name in enumerate(retval['action_names']):
    data = data_by_pose[action_name]

    for camera_idx, camera_name in enumerate(retval['camera_names']):
        if DEBUG: 
            print(f"{action_name}, cam {camera_name}: ({action_idx},{camera_idx})")

        cam_retval = retval['cameras'][action_idx][camera_idx]
        camera_data = data['camera_data'][camera_name]

        # TODO: Check if need transpose
        cam_retval['R'] = np.array(camera_data['R'])
        cam_retval['K'] = np.array(camera_data['K'])
        cam_retval['t'] = np.array(camera_data['t'])
        cam_retval['dist']=np.array(camera_data['dist'])

    if DEBUG:
        print("")

    for frame_idx, frame_name in enumerate(data['valid_frames']):
        table_segment = np.empty(len(data['valid_frames']), dtype=table_dtype)

        # TODO: Poses changing from CMU to H36M, if the current one doesn't do it automatically
        person_data_arr = data['person_data'][frame_name]

        for person_data in person_data_arr:
            if DEBUG:
                print(
                    f"{action_name}, frame {frame_name}, person {person_data['id']}"
                )

            table_segment['person_id'] = person_data['id']
            table_segment['action_idx'] = action_idx 
            table_segment['frame_names'] = np.array(data['valid_frames']).astype(np.int16)  # TODO: Check this
            table_segment['keypoints'] = person_data['joints']

            # Load BBOX Data (loaded above from external MRCNN Detections file)
            # let a (0,0,0,0) bbox mean that this view is missing
            table_segment['bbox_by_camera_tlbr'] = 0

            for camera_idx, camera_name in enumerate(retval['camera_names']):
                for bbox, frame_idx in zip(table_segment['bbox_by_camera_tlbr'], data['valid_frames']):
                    bbox[camera_idx] = bbox_data[action_name][camera_name][frame_idx]

            retval['table'].append(table_segment)

    if DEBUG: 
        print("\n")

# Check 
if DEBUG:
    for action_idx, action_name in enumerate(retval['action_names']):
        for camera_idx, camera_name in enumerate(retval['camera_names']):
            print(data_by_pose[retval['action_names'][action_idx]]
                ["camera_data"][retval['camera_names'][camera_idx]]['R'])

            print(retval['cameras'][action_idx][camera_idx]['R'])
            print("")

# NOTE: Camera data also need to be filled 
# camera_id = int(camera_name.replace('_'$1', "'))

# Ready to Save!
retval['table'] = np.concatenate(retval['table'])
assert retval['table'].ndim == 1

print('Total frames in CMU Panoptic Dataset:', len(retval['table']))

print("\nSaving labels file...")

try:
    np.save(destination_file_path, retval)
    print(f"Labels file saved to {destination_file_path}")
except: 
    raise f"Failed to save file {destination_file_path}"
