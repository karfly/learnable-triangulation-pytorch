from copy import deepcopy
import numpy as np
import pickle
import random

from scipy.optimize import least_squares

import torch
from torch import nn

from mvn.utils import op, multiview, img, misc, volumetric

from mvn.models import pose_resnet
from mvn.models.v2v import V2VModel


class RANSACTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)
        
        self.direct_optimization = config.model.direct_optimization

    def forward(self, images, proj_matricies, batch):
        batch_size, n_views = images.shape[:2]

        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])

        # forward backbone and integrate
        heatmaps, _, _, _ = self.backbone(images)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])

        # calculate shapes
        image_shape = tuple(images.shape[3:])
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])

        # keypoints 2d
        _, max_indicies = torch.max(heatmaps.view(batch_size, n_views, n_joints, -1), dim=-1)
        keypoints_2d = torch.stack([max_indicies % heatmap_shape[1], max_indicies // heatmap_shape[1]], dim=-1).to(images.device)

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])
        keypoints_2d = keypoints_2d_transformed

        # triangulate (cpu)
        keypoints_2d_np = keypoints_2d.detach().cpu().numpy()
        proj_matricies_np = proj_matricies.detach().cpu().numpy()

        keypoints_3d = np.zeros((batch_size, n_joints, 3))
        confidences = np.zeros((batch_size, n_views, n_joints))  # plug
        for batch_i in range(batch_size):
            for joint_i in range(n_joints):
                current_proj_matricies = proj_matricies_np[batch_i]
                points = keypoints_2d_np[batch_i, :, joint_i]
                keypoint_3d, _ = self.triangulate_ransac(current_proj_matricies, points, direct_optimization=self.direct_optimization)
                keypoints_3d[batch_i, joint_i] = keypoint_3d

        keypoints_3d = torch.from_numpy(keypoints_3d).type(torch.float).to(images.device)
        confidences = torch.from_numpy(confidences).type(torch.float).to(images.device)

        return keypoints_3d, keypoints_2d, heatmaps, confidences

    def triangulate_ransac(self, proj_matricies, points, n_iters=10, reprojection_error_epsilon=15, direct_optimization=True):
        assert len(proj_matricies) == len(points)
        assert len(points) >= 2

        proj_matricies = np.array(proj_matricies)
        points = np.array(points)

        n_views = len(points)

        # determine inliers
        view_set = set(range(n_views))
        inlier_set = set()
        for i in range(n_iters):
            sampled_views = sorted(random.sample(view_set, 2))

            keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(proj_matricies[sampled_views], points[sampled_views])
            reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), points, proj_matricies)[0]

            new_inlier_set = set(sampled_views)
            for view in view_set:
                current_reprojection_error = reprojection_error_vector[view]
                if current_reprojection_error < reprojection_error_epsilon:
                    new_inlier_set.add(view)

            if len(new_inlier_set) > len(inlier_set):
                inlier_set = new_inlier_set

        # triangulate using inlier_set
        if len(inlier_set) == 0:
            inlier_set = view_set.copy()

        inlier_list = np.array(sorted(inlier_set))
        inlier_proj_matricies = proj_matricies[inlier_list]
        inlier_points = points[inlier_list]

        keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(inlier_proj_matricies, inlier_points)
        reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
        reprojection_error_mean = np.mean(reprojection_error_vector)

        keypoint_3d_in_base_camera_before_direct_optimization = keypoint_3d_in_base_camera
        reprojection_error_before_direct_optimization = reprojection_error_mean

        # direct reprojection error minimization
        if direct_optimization:
            def residual_function(x):
                reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([x]), inlier_points, inlier_proj_matricies)[0]
                residuals = reprojection_error_vector
                return residuals

            x_0 = np.array(keypoint_3d_in_base_camera)
            res = least_squares(residual_function, x_0, loss='huber', method='trf')

            keypoint_3d_in_base_camera = res.x
            reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
            reprojection_error_mean = np.mean(reprojection_error_vector)

        return keypoint_3d_in_base_camera, inlier_list


class AlgebraicTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.use_confidences = config.model.use_confidences

        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False

        if self.use_confidences:
            config.model.backbone.alg_confidences = True

        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)

        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier


    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])

        # forward backbone and integral
        if self.use_confidences:
            heatmaps, _, alg_confidences, _ = self.backbone(images)
        else:
            heatmaps, _, _, _ = self.backbone(images)
            alg_confidences = torch.ones(batch_size * n_views, heatmaps.shape[1]).type(torch.float).to(device)

        heatmaps_before_softmax = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        keypoints_2d, heatmaps = op.integrate_tensor_2d(heatmaps * self.heatmap_multiplier, self.heatmap_softmax)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        keypoints_2d = keypoints_2d.view(batch_size, n_views, *keypoints_2d.shape[1:])
        alg_confidences = alg_confidences.view(batch_size, n_views, *alg_confidences.shape[1:])

        # norm confidences
        alg_confidences = alg_confidences / alg_confidences.sum(dim=1, keepdim=True)
        alg_confidences = alg_confidences + 1e-5  # for numerical stability

        # calculate shapes
        image_shape = tuple(images.shape[3:])
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])
        keypoints_2d = keypoints_2d_transformed

        # triangulate
        try:
            keypoints_3d = multiview.triangulate_batch_of_points(
                proj_matricies, keypoints_2d,
                confidences_batch=alg_confidences
            )
        except RuntimeError as e:
            print("Error: ", e)

            print("confidences =", confidences_batch_pred)
            print("proj_matricies = ", proj_matricies)
            print("keypoints_2d_batch_pred =", keypoints_2d_batch_pred)
            exit()

        return keypoints_3d, keypoints_2d, heatmaps, alg_confidences


class VolumetricTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_aggregation_method

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size

        self.cuboid_side = config.model.cuboid_side

        self.kind = config.model.kind
        self.use_gt_pelvis = config.model.use_gt_pelvis

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        # modules
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)

        for p in self.backbone.final_layer.parameters():
            p.requires_grad = False

        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )

        self.volume_net = V2VModel(32, self.num_joints)


    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        heatmaps, features, _, vol_confidences = self.backbone(images)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        n_joints = heatmaps.shape[2]

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, heatmap_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        proj_matricies = proj_matricies.float().to(device)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            # if self.use_precalculated_pelvis:
            if self.use_gt_pelvis:
                keypoints_3d = batch['keypoints_3d'][batch_i]
            else:
                keypoints_3d = batch['pred_keypoints_3d'][batch_i]

            if self.kind == "coco":
                base_point = (keypoints_3d[11, :3] + keypoints_3d[12, :3]) / 2
            elif self.kind == "mpii":
                base_point = keypoints_3d[6, :3]

            base_points[batch_i] = torch.from_numpy(base_point).to(device)

            # build cuboid L x L x L
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            if self.kind == "coco":
                axis = [0, 1, 0]  # y axis
            elif self.kind == "mpii":
                axis = [0, 0, 1]  # z axis

            center = torch.from_numpy(base_point).type(torch.float).to(device)

            # rotate
            coord_volume = coord_volume - center
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
            coord_volume = coord_volume + center

            # transfer
            if self.transfer_cmu_to_human36m:  # different world coordinates
                coord_volume = coord_volume.permute(0, 2, 1, 3)
                inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
                coord_volume = coord_volume.index_select(1, inv_idx)

            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])

        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

        # integral 3d (V2V)
        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return vol_keypoints_3d, features, volumes, vol_confidences, cuboids, coord_volumes, base_points
