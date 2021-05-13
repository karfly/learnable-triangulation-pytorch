import os
import time
from collections import defaultdict

import numpy as np
import cv2

from torch.utils.data import Dataset

from mvn.utils.multiview import Camera, build_intrinsics
from mvn.utils.img import resize_image, crop_image, normalize_image, scale_bbox, make_with_target_intrinsics, rotation_matrix_from_vectors

TARGET_INTRINSICS = build_intrinsics(
    translation=(680, 690),
    f=(1.6e3, 1.6e3),
    shear=0
)

class Human36MMultiViewDataset(Dataset):
    """
        Human3.6M for multiview tasks.
    """
    def __init__(self,
                 h36m_root,
                 labels_path,
                 pred_results_path=None,
                 image_shape=(256, 256),
                 train=False,
                 test=False,
                 retain_every_n_frames_in_train=100,
                 retain_every_n_frames_in_test=100,
                 with_damaged_actions=False,
                 cuboid_side=2000.0,
                 scale_bbox=1.5,
                 norm_image=True,
                 kind="mpii",
                 undistort_images=False,
                 ignore_cameras=[],
                 crop=True,
                 resample=False,
                 look_at_pelvis=False
                 ):
        """
            h36m_root:
                Path to 'processed/' directory in Human3.6M
            labels_path:
                Path to 'human36m-multiview-labels.npy' generated by 'generate-labels-npy-multiview.py'
                from https://github.sec.samsung.net/RRU8-VIOLET/human36m-preprocessing
            retain_every_n_frames_in_test:
                By default, there are 159 181 frames in training set and 26 634 in test (val) set.
                With this parameter, test set frames will be evenly skipped frames so that the
                test set size is `26634 // retain_every_n_frames_test`.
                Use a value of 13 to get 2049 frames in test set.
            with_damaged_actions:
                If `True`, will include 'S9/[Greeting-2,SittingDown-2,Waiting-1]' in test set.
            kind:
                Keypoint format, 'mpii' or 'human36m'
            ignore_cameras:
                A list with indices of cameras to exclude (0 to 3 inclusive)
        """
        assert train or test, '`Human36MMultiViewDataset` must be constructed with at least one of `test=True` / `train=True`'
        assert kind in ("mpii", "human36m")

        self.h36m_root = h36m_root
        self.labels_path = labels_path
        self.image_shape = None if image_shape is None else tuple(image_shape)
        self.scale_bbox = scale_bbox
        self.norm_image = norm_image
        self.cuboid_side = cuboid_side
        self.kind = kind
        self.undistort_images = undistort_images
        self.ignore_cameras = ignore_cameras
        self.crop = crop
        self.do_resample = resample
        self.look_at_pelvis = look_at_pelvis

        self.labels = np.load(labels_path, allow_pickle=True).item()

        n_cameras = len(self.labels['camera_names'])
        assert all(
            camera_idx in range(n_cameras)
            for camera_idx in self.ignore_cameras
        )

        train_subjects = [
            self.labels['subject_names'].index(x)
            for x in ['S1', 'S6', 'S7', 'S8']  # todo solve missings in 'S5'
        ]
        test_subjects = [
            self.labels['subject_names'].index(x)
            for x in ['S9', 'S11']
        ]

        indices = []

        if train:
            mask = np.isin(
                self.labels['table']['subject_idx'], train_subjects, assume_unique=True
            )
            indices.append(
                np.nonzero(mask)[0][::retain_every_n_frames_in_train]
            )
        else:  # test
            mask = np.isin(self.labels['table']['subject_idx'], test_subjects, assume_unique=True)

            if not with_damaged_actions:
                mask_S9 = self.labels['table']['subject_idx'] == self.labels['subject_names'].index('S9')

                damaged_actions = [
                    self.labels['action_names'].index(x)
                    for x in ['Greeting-2', 'SittingDown-2', 'Waiting-1']
                ]
                mask_damaged_actions = np.isin(self.labels['table']['action_idx'], damaged_actions)

                mask &= ~(mask_S9 & mask_damaged_actions)

            indices.append(np.nonzero(mask)[0][::retain_every_n_frames_in_test])

        self.labels['table'] = self.labels['table'][np.concatenate(indices)]
        self.indices = indices

        self.num_keypoints = 16 if kind == "mpii" else 17
        assert self.labels['table']['keypoints'].shape[1] == 17, "Use a newer 'labels' file"

        self.keypoints_3d_pred = None
        if pred_results_path is not None:
            pred_results = np.load(pred_results_path, allow_pickle=True)
            keypoints_3d_pred = pred_results['keypoints_3d'][np.argsort(pred_results['indexes'])]
            self.keypoints_3d_pred = keypoints_3d_pred[::retain_every_n_frames_in_test]

            assert len(self.keypoints_3d_pred) == len(self), \
                f"[train={train}, test={test}] {labels_path} has {len(self)} samples, but '{pred_results_path}' " + \
                f"has {len(self.keypoints_3d_pred)}. Did you follow all preprocessing instructions carefully?"

        self.meshgrids = None

    def __len__(self):
        return len(self.labels['table'])

    def _get_frame_info(self, idx):
        shot = self.labels['table'][idx]

        subject_idx = shot['subject_idx']
        action_idx = shot['action_idx']

        subject = self.labels['subject_names'][subject_idx]
        action = self.labels['action_names'][action_idx]
        frame_idx = shot['frame_idx']  # unique ID in all dataset

        return {
            'shot': shot,
            'subject': subject,
            'action': action,
            'frame idx': frame_idx
        }

    def _get_view_path(self, subject_name, action_name, camera_name, frame_idx):
        return os.path.join(
            self.h36m_root,
            subject_name,
            action_name,
            'imageSequence' + '-undistorted' * self.undistort_images,
            camera_name,
            'img_%06d.jpg' % (frame_idx + 1)
        )

    def _get_view_path_from_shot(self, shot, camera_name):
        subject_idx = shot['subject_idx']
        action_idx = shot['action_idx']

        subject_name = self.labels['subject_names'][subject_idx]
        action_name = self.labels['action_names'][action_idx]
        frame_idx = shot['frame_idx']

        return self._get_view_path(
            subject_name, action_name, camera_name, frame_idx
        )

    def _load_image(self, subject, action, camera_name, frame_idx, sleep_minutes=5, max_attempts=10):
        image_path = self._get_view_path(
            subject, action, camera_name, frame_idx
        )

        image = cv2.imread(image_path)

        curr_attempt = 0
        while curr_attempt < max_attempts:
            curr_attempt += 1
            if not (image is None):
                return image

            print('failed loading {} from disk for the # {} time'.format(
                image_path,
                curr_attempt
            ))

            if curr_attempt < max_attempts:  # will re-do cycle
                print('=> sleeping {} minutes ...'.format(sleep_minutes))
                time.sleep(sleep_minutes * 60)
                print(
                    '... AWAKE! retrying (for the # {} time)'.format(
                        curr_attempt + 1
                    )
                )

        raise IOError('fix that cluster !!!')

    def __getitem__(self, idx):
        sample = defaultdict(list)  # return value
        shot = self.labels['table'][idx]

        subject_idx = shot['subject_idx']
        action_idx = shot['action_idx']

        subject = self.labels['subject_names'][subject_idx]
        action = self.labels['action_names'][action_idx]
        frame_idx = shot['frame_idx']

        for camera_idx, camera_name in enumerate(self.labels['camera_names']):
            if camera_idx in self.ignore_cameras:
                continue

            # load bounding box
            bbox = shot['bbox_by_camera_tlbr'][camera_idx][[1, 0, 3, 2]]  # TLBR to LTRB
            bbox_height = bbox[2] - bbox[0]
            if bbox_height == 0:
                # convention: if the bbox is empty, then this view is missing
                continue

            # scale the bounding box
            bbox = scale_bbox(bbox, self.scale_bbox)

            image = self._load_image(subject, action, camera_name, frame_idx)

            # load camera
            shot_camera = self.labels['cameras'][shot['subject_idx'], camera_idx]
            retval_camera = Camera(
                shot_camera['R'],
                shot_camera['t'],
                shot_camera['K'],
                shot_camera['dist'],
                camera_name
            )

            if self.crop:  # crop image
                image = crop_image(image, bbox)
                retval_camera.update_after_crop(bbox)

            if self.do_resample:
                square = (0, 0, 1000, 1000)  # have same size
                image = crop_image(image, square)
                retval_camera.update_after_crop(square)

                # todo get rid of 1k + eps
                # have same intrinsics
                new_shape, cropping_box = make_with_target_intrinsics(
                    image,
                    retval_camera.K,
                    TARGET_INTRINSICS
                )
                image_shape_before_resize = image.shape[:2]
                image = resize_image(image, new_shape)
                retval_camera.update_after_resize(
                    image_shape_before_resize, new_shape
                )

                image = crop_image(image, cropping_box)
                retval_camera.update_after_crop(cropping_box)

            if self.look_at_pelvis:
                kp_in_world = shot['keypoints'][:self.num_keypoints]
                kp_in_cam = retval_camera.world2cam()(kp_in_world)

                pelvis_index = 6  # H36M dataset, not CMU
                pelvis_vector = kp_in_cam[pelvis_index, ...]

                # ... => find rotation matrix pelvis to z ...
                z_axis = [0, 0, 1]
                Rt = rotation_matrix_from_vectors(pelvis_vector, z_axis)

                # ... and update E <- R.dot(E)
                retval_camera.update_roto_extrsinsics(Rt)

            if self.image_shape is not None:  # resize
                image_shape_before_resize = image.shape[:2]
                image = resize_image(image, self.image_shape)
                retval_camera.update_after_resize(
                    image_shape_before_resize, self.image_shape
                )

            if self.norm_image:
                image = normalize_image(image)

            sample['images'].append(image)
            # sample['detections'].append(bbox + (1.0,))  # TODO add real confidences
            sample['cameras'].append(retval_camera)
            sample['proj_matrices'].append(retval_camera.projection)

        if self.meshgrids:  # undistort, call `self.make_meshgrids` beforehand
            available_cameras = list(range(len(self.labels['action_names'])))
            for camera_idx, bbox in enumerate(shot['bbox_by_camera_tlbr']):
                if bbox[2] == bbox[0]:  # bbox is empty, which means that this camera is missing
                    available_cameras.remove(camera_idx)

            for i, (camera_idx, image) in enumerate(zip(available_cameras, sample['images'])):
                meshgrid_int16 = self.meshgrids[shot['subject_idx'], camera_idx]
                image_undistorted = cv2.remap(image, *meshgrid_int16, cv2.INTER_CUBIC)
                sample['images'][i] = image_undistorted

        # 3D keypoints
        # add dummy confidences
        sample['keypoints_3d'] = np.pad(
            shot['keypoints'][:self.num_keypoints],
            ((0,0), (0,1)), 'constant', constant_values=1.0
        )

        # build cuboid
        # base_point = sample['keypoints_3d'][6, :3]
        # sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
        # position = base_point - sides / 2
        # sample['cuboids'] = volumetric.Cuboid3D(position, sides)

        # save sample's index
        sample['indexes'] = idx

        if self.keypoints_3d_pred is not None:
            sample['pred_keypoints_3d'] = self.keypoints_3d_pred[idx]

        sample.default_factory = None
        return sample

    def make_meshgrids(self):
        n_subjects = len(self.labels['subject_names'])
        n_cameras = len(self.labels['camera_names'])
        meshgrids = np.empty((n_subjects, n_cameras), dtype=object)

        for sample_idx in range(len(self.labels['table'])):
            subject_idx = self.labels['table']['subject_idx'][sample_idx]
            
            if not meshgrids[subject_idx].any():
                bboxes = self.labels['table']['bbox_by_camera_tlbr'][sample_idx]
            
                if (bboxes[:, 2] - bboxes[:, 0]).min() > 0:  # if == 0, then some camera is missing
                    sample = self.__getitem__(sample_idx)
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

        return meshgrids

    def evaluate_using_per_pose_error(self, per_pose_error, split_by_subject):
        def evaluate_by_actions(self, per_pose_error, mask=None):
            if mask is None:
                mask = np.ones_like(per_pose_error, dtype=bool)

            action_scores = {
                'Average': {'total_loss': per_pose_error[mask].sum(), 'frame_count': np.count_nonzero(mask)}
            }

            for action_idx in range(len(self.labels['action_names'])):
                action_mask = (self.labels['table']['action_idx'] == action_idx) & mask
                action_per_pose_error = per_pose_error[action_mask]
                action_scores[self.labels['action_names'][action_idx]] = {
                    'total_loss': action_per_pose_error.sum(), 'frame_count': len(action_per_pose_error)
                }

            action_names_without_trials = \
                [name[:-2] for name in self.labels['action_names'] if name.endswith('-1')]

            for action_name_without_trial in action_names_without_trials:
                combined_score = {'total_loss': 0.0, 'frame_count': 0}

                for trial in 1, 2:
                    action_name = '%s-%d' % (action_name_without_trial, trial)
                    combined_score['total_loss' ] += action_scores[action_name]['total_loss']
                    combined_score['frame_count'] += action_scores[action_name]['frame_count']
                    del action_scores[action_name]

                action_scores[action_name_without_trial] = combined_score

            for k, v in action_scores.items():
                action_scores[k] = float('nan') if v['frame_count'] == 0 else (v['total_loss'] / v['frame_count'])

            return action_scores

        subject_scores = {
            'Average': evaluate_by_actions(self, per_pose_error)
        }

        for subject_idx in range(len(self.labels['subject_names'])):
            subject_mask = self.labels['table']['subject_idx'] == subject_idx
            subject_scores[self.labels['subject_names'][subject_idx]] = \
                evaluate_by_actions(self, per_pose_error, subject_mask)

        return subject_scores

    def evaluate(self, keypoints_3d_predicted, indices_predicted=None, split_by_subject=False, transfer_cmu_to_human36m=False, transfer_human36m_to_human36m=False):
        keypoints_gt = self.labels['table']['keypoints'][:, :self.num_keypoints]

        if not (indices_predicted is None):
            keypoints_gt = keypoints_gt[indices_predicted]

        if keypoints_3d_predicted.shape != keypoints_gt.shape:
            raise ValueError(
                '`keypoints_3d_predicted` shape should be %s, got %s' % \
                (keypoints_gt.shape, keypoints_3d_predicted.shape))

        if transfer_cmu_to_human36m or transfer_human36m_to_human36m:
            human36m_joints = [10, 11, 15, 14, 1, 4]
            if transfer_human36m_to_human36m:
                cmu_joints = [10, 11, 15, 14, 1, 4]
            else:
                cmu_joints = [10, 8, 9, 7, 14, 13]

            keypoints_gt = keypoints_gt[:, human36m_joints]
            keypoints_3d_predicted = keypoints_3d_predicted[:, cmu_joints]

        # mean error per 16/17 joints in mm, for each pose
        per_pose_error = np.sqrt(((keypoints_gt - keypoints_3d_predicted) ** 2).sum(2)).mean(1)

        # relative (to the pelvis) mean error per 16/17 joints in mm, for each pose
        if not (transfer_cmu_to_human36m or transfer_human36m_to_human36m):
            root_index = 6 if self.kind == "mpii" else 6  # the pelvis
        else:
            root_index = 0  # the pelvis

        keypoints_gt_relative = keypoints_gt - keypoints_gt[:, root_index:root_index + 1, :]
        keypoints_3d_predicted_relative = keypoints_3d_predicted - keypoints_3d_predicted[:, root_index:root_index + 1, :]

        per_pose_error_relative = np.sqrt(((keypoints_gt_relative - keypoints_3d_predicted_relative) ** 2).sum(2)).mean(1)  # should be (alg, vol) = (22.6, 20.8), excluding 'with_damaged_actions' frames

        result = {
            'per_pose_error': self.evaluate_using_per_pose_error(per_pose_error, split_by_subject),
            'per_pose_error_relative': self.evaluate_using_per_pose_error(per_pose_error_relative, split_by_subject)
        }

        return result['per_pose_error_relative']['Average']['Average'], result
