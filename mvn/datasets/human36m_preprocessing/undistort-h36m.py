"""
    Undistort images in Human3.6M and save them alongside (in ".../imageSequence-undistorted/...").

    Usage: `python3 undistort-h36m.py <path/to/Human3.6M-root> <path/to/human36m-multiview-labels.npy> <num-processes>`
"""
import torch
import numpy as np
import cv2
from tqdm import tqdm

import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../.."))
from mvn.datasets.human36m import Human36MMultiViewDataset

h36m_root = os.path.join(sys.argv[1], "processed")
labels_multiview_npy_path = sys.argv[2]
number_of_processes = int(sys.argv[3])

dataset = Human36MMultiViewDataset(
    h36m_root,
    labels_multiview_npy_path,
    train=True,                       # include all possible data
    test=True,
    image_shape=None,                 # don't resize
    retain_every_n_frames_in_test=1,  # yes actually ALL possible data
    with_damaged_actions=True,        # I said ALL DATA
    kind="mpii",
    norm_image=False,                 # don't do unnecessary image processing
    crop=False)                       # don't crop
print("Dataset length:", len(dataset))

n_subjects = len(dataset.labels['subject_names'])
n_cameras = len(dataset.labels['camera_names'])

# First, prepare: compute distorted meshgrids
print("Computing distorted meshgrids")
meshgrids = np.empty((n_subjects, n_cameras), dtype=object)

for sample_idx in range(len(dataset))):
    subject_idx = dataset.labels['table']['subject_idx'][sample_idx]
    
    if not meshgrids[subject_idx].any():
        bboxes = dataset.labels['table']['bbox_by_camera_tlbr'][sample_idx]
    
        if (bboxes[:, 2] - bboxes[:, 0]).min() > 0: # if == 0, then some camera is missing
            sample = dataset[sample_idx]
            assert len(sample['images']) == n_cameras
    
            for camera_idx, (camera, image) in enumerate(zip(sample['cameras'], sample['images'])):
                h, w = image.shape[:2]
                
                fx, fy = camera.K[0, 0], camera.K[1, 1]
                cx, cy = camera.K[0, 2], camera.K[1, 2]
                
                grid_x = (np.arange(w, dtype=np.float32) - cx) / fx
                grid_y = (np.arange(h, dtype=np.float32) - cy) / fy
                meshgrid = np.stack(np.meshgrid(grid_x, grid_y), axis=2).reshape(-1, 2)

                # distort meshgrid points
                k = camera.dist[:3].copy(); k[2] = camera.dist[-1]
                p = camera.dist[2:4].copy()
                
                r2 = meshgrid[:, 0] ** 2 + meshgrid[:, 1] ** 2
                radial = meshgrid * (1 + k[0] * r2 + k[1] * r2**2 + k[2] * r2**3).reshape(-1, 1)
                tangential_1 = p.reshape(1, 2) * np.broadcast_to(meshgrid[:, 0:1] * meshgrid[:, 1:2], (len(meshgrid), 2))
                tangential_2 = p[::-1].reshape(1, 2) * (meshgrid**2 + np.broadcast_to(r2.reshape(-1, 1), (len(meshgrid), 2)))

                meshgrid = radial + tangential_1 + tangential_2

                # move back to screen coordinates
                meshgrid *= np.array([fx, fy]).reshape(1, 2)
                meshgrid += np.array([cx, cy]).reshape(1, 2)

                # cache (save) distortion maps
                meshgrids[subject_idx, camera_idx] = cv2.convertMaps(meshgrid.reshape((h, w, 2)), None, cv2.CV_16SC2)

# Now the main part: undistort images
def undistort_and_save(idx):
    sample = dataset[idx]
    
    shot = dataset.labels['table'][idx]
    subject_idx = shot['subject_idx']
    action_idx = shot['action_idx']
    frame_idx = shot['frame_idx']

    subject = dataset.labels['subject_names'][subject_idx]
    action = dataset.labels['action_names'][action_idx]

    available_cameras = list(range(len(dataset.labels['action_names'])))
    for camera_idx, bbox in enumerate(shot['bbox_by_camera_tlbr']):
        if bbox[2] == bbox[0]: # bbox is empty, which means that this camera is missing
            available_cameras.remove(camera_idx)

    for camera_idx, image in zip(available_cameras, sample['images']):
        camera_name = dataset.labels['camera_names'][camera_idx]

        output_image_folder = os.path.join(
            h36m_root, subject, action, 'imageSequence-undistorted', camera_name)
        output_image_path = os.path.join(output_image_folder, 'img_%06d.jpg' % (frame_idx+1))
        os.makedirs(output_image_folder, exist_ok=True)

        meshgrid_int16 = meshgrids[subject_idx, camera_idx]
        image_undistorted = cv2.remap(image, *meshgrid_int16, cv2.INTER_CUBIC)

        cv2.imwrite(output_image_path, image_undistorted)

print(f"Undistorting images using {number_of_processes} parallel processes")
cv2.setNumThreads(1)
import multiprocessing

pool = multiprocessing.Pool(number_of_processes)
for _ in tqdm(pool.imap_unordered(
    undistort_and_save, range(len(dataset)), chunksize=10), total=len(dataset)):
    pass

pool.close()
pool.join()
