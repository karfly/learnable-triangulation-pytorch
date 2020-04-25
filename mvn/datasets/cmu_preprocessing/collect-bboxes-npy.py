#!/bin/python

'''
Collate BBOX data from CMU Panoptic Dataset into a npy file
'''
USAGE_PROMPT="""
$ python3 collect-bboxes-npy.py <path/to/mrcnn-detections/folder> <path/to/output/file(optional)>

Example:
$ python3 collect-bboxes-npy.py $THIS_REPOSITORY/data/pretrained/cmu/mrcnn-detections $THIS_REPOSITORY/mvn/datasets/cmu_preprocessing
"""

try:
    bbox_dir = sys.argv[1]
except:
    print(USAGE_PROMPT)
    exit()

try:
    output_dir = sys.argv[2]
except:
    output_dir = "./"

output_file = os.path.join(output_dir, "cmu-bboxes.npy")

# BBOX Data
bbox_data = {}
#bbox_dir = os.path.join(bbox_root, 'mrcnn-detections')

assert os.path.isdir(bbox_dir), "Invalid BBOX directory '%s'" % bbox_dir

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

        bbox_data[action_name][camera_name] = bbox_data_arr

        if DEBUG:
            print(action_name, camera_name)
        
return bbox_data
