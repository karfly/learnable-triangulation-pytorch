import numpy as np
import torch

from mvn.utils.img import image_batch_to_torch

def make_collate_fn(randomize_n_views=True, min_n_views=10, max_n_views=31):
    def collate_fn(items):
        items = list(filter(lambda x: x is not None, items))
        if len(items) == 0:
            print("All items in batch are None")
            return None

        batch = dict()
        total_n_views = min(len(item['images']) for item in items)

        indexes = np.arange(total_n_views)
        if randomize_n_views:
            n_views = np.random.randint(min_n_views, min(total_n_views, max_n_views) + 1)
            indexes = np.random.choice(np.arange(total_n_views), size=n_views, replace=False)
        else:
            indexes = np.arange(total_n_views)

        batch['images'] = np.stack([
            np.stack([
                item['images'][i]
                for item in items
            ], axis=0)
            for i in indexes
        ], axis=0).swapaxes(0, 1)
        batch['detections'] = np.array(
            [
                [
                    item['detections'][i] for item in items
                ]
                for i in indexes
            ]
        ).swapaxes(0, 1)
        batch['cameras'] = [
            [
                item['cameras'][i]
                for item in items
            ]
            for i in indexes
        ]

        batch['keypoints_3d'] = [item['keypoints_3d'] for item in items]
        # batch['cuboids'] = [item['cuboids'] for item in items]
        batch['indexes'] = [
            item['indexes']
            for item in items
        ]

        try:
            batch['pred_keypoints_3d'] = np.array([
                item['pred_keypoints_3d']
                for item in items
            ])
        except:
            pass

        return batch

    return collate_fn


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def cam2cam_batch(src, target, cameras_batch, batch_i):
    if src == target:  # just apply intrinsics
        cam = cameras_batch[src][batch_i]  # multiview.Camera
        return cam.intrinsics_padded  # 3 x 4

    src = cameras_batch[src][batch_i]
    target = cameras_batch[target][batch_i]
    return target.intrinsics_padded.dot(
        target.extrinsics_padded
    ).dot(
        np.linalg.inv(src.extrinsics_padded)
    )  # 3 x 4


def cam2cam_precomputed_batch(src, target, cameras_batch, batch_i, cam2cams):
    if src == target:  # just apply intrinsics
        cam = cameras_batch[src][batch_i]  # multiview.Camera
        return cam.intrinsics_padded  # 3 x 4

    return cameras_batch[target][batch_i].intrinsics_padded.dot(
        cam2cams[batch_i][target]  # cam2cam from src -> target
    )  # 3 x 4


def prepare_batch(batch, device, config, is_train=True):
    images_batch = []  # images
    for image_batch in batch['images']:
        image_batch = image_batch_to_torch(image_batch)
        image_batch = image_batch.to(device)
        images_batch.append(image_batch)

    images_batch = torch.stack(images_batch, dim=0)
    batch_size, n_views = images_batch.shape[0], images_batch.shape[1]

    # 3D keypoints
    keypoints_3d_batch_gt = torch.from_numpy(
        np.stack(batch['keypoints_3d'], axis=0)[:, :, :3]
    ).float().to(device)

    # 3D keypoints validity
    keypoints_3d_validity_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[:, :, 3:]).float().to(device)

    # projection matricies
    proj_matricies_batch = torch.stack([
        torch.stack([
            torch.from_numpy(camera.projection)
            for camera in camera_batch
        ], dim=0)
        for camera_batch in batch['cameras']
    ], dim=0).transpose(1, 0)  # shape (batch_size=8, n_views=4, 3, 4)
    proj_matricies_batch = proj_matricies_batch.float().to(device)

    return images_batch, keypoints_3d_batch_gt, keypoints_3d_validity_batch_gt, proj_matricies_batch
