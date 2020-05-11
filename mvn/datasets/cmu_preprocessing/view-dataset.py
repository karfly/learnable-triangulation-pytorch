#!/bin/python

"""
    A GUI script for inspecting Human3.6M and its wrappers, namely:
    - `CMUPanopticDataset`
    - `human36m-multiview-labels-**bboxes.npy`

    Usage: `python3 view-dataset.py <path/to/Human3.6M-root> <path/to/human36m-multiview-labels-*bboxes.npy> [<start-sample-number> [<samples-per-step>]]
"""
import torch
import numpy as np
import cv2

import os, sys
import math

cmu_root = sys.argv[1]
labels_path = sys.argv[2]

try:    sample_idx = int(sys.argv[3])
except: sample_idx = 0

try:    step = int(sys.argv[4])
except: step = 10

try:
    imgdir = "dataset_imgs"
    save_images_instead = (int(sys.argv[5]) == 1)
    print(f"Saving images to {imgdir} instead of displaying them...")
except: 
    save_images_instead = False

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../.."))
from mvn.datasets.cmupanoptic import CMUPanopticDataset

dataset = CMUPanopticDataset(
    cmu_root,
    labels_path,
    train=True,
    test=True,
    image_shape=(512,512),
    retain_every_n_frames_in_test=1,
    scale_bbox=1.0,
    kind='cmu',
    norm_image=False,
    ignore_cameras=[])

print("Total Samples:", len(dataset))
print("Total Images Shown (Est.):", math.ceil(len(dataset)/step))

prev_action = None
patience = 0

while sample_idx < len(dataset):
    sample = dataset[sample_idx]

    camera_idx = 0
    image = sample['images'][camera_idx]
    camera = sample['cameras'][camera_idx]

    display = image.copy()

    from mvn.utils.multiview import project_3d_points_to_image_plane_without_distortion as project
    keypoints_2d = project(camera.projection, sample['keypoints_3d'][:, :3])
    
    for i,(x,y) in enumerate(keypoints_2d):
        cv2.circle(display, (int(x), int(y)), 3, (0,0,255), -1)
        # cv2.putText(display, str(i), (int(x)+3, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
    
    # Get window name
    sample_info = dataset.labels['table'][sample_idx]
    #subject_name = dataset.labels['subject_names'][sample_info['subject_idx']]
    person_id = sample_info['person_id']
    action_name = dataset.labels['action_names'][sample_info['action_idx']]
    camera_name = dataset.labels['camera_names'][camera_idx]
    frame_idx = sample_info['frame_name']

    
    if save_images_instead:
        img_path = os.path.join('.', imgdir, action_name, camera_name)
        if not os.path.isdir(img_path):
            os.mkdir(img_path)

        img_path = os.path.join(img_path, f"{frame_idx:08}_p{person_id}.jpg")
        
        try: 
            print(f"Saving image to {img_path}")
            cv2.imwrite(img_path, display)
        except:
            print(f"Error: Cannot save to {img_path}")
    else:
        title = f"Person {person_id}: {action_name}/{camera_name}/{frame_idx}"

        cv2.imshow('w', display)
        cv2.setWindowTitle('w', title)
        c = cv2.waitKey(0) % 256

        if c == ord('q') or c == 27:
            print('Quitting...')
            break

    action = sample_info['action_idx']
    if action != prev_action: # started a new action
        prev_action = action
        patience = 2000
        sample_idx += step
    elif patience == 0: # an action ended, jump to the start of new action
        while True:
            sample_idx += step
            action = dataset.labels['table'][sample_idx]['action_idx']
            if action != prev_action:
                break
    else: # in progess, just increment sample_idx
        patience -= 1
        sample_idx += step

print("Done.")
