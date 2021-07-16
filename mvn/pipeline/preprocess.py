import torch


def center2pelvis(keypoints_2d, pelvis_i):
    """ pelvis -> (0, 0) """

    n_joints = keypoints_2d.shape[2]
    pelvis_point = keypoints_2d[:, :, pelvis_i, :]

    return keypoints_2d - pelvis_point.unsqueeze(2).repeat(1, 1, n_joints, 1)  # in each view: joint coords - pelvis coords


def dist2pelvis(keypoints_2d_in_view, pelvis_i):
    return torch.mean(torch.cat([
        torch.norm(
            keypoints_2d_in_view[i] -\
            keypoints_2d_in_view[pelvis_i]
        ).unsqueeze(0)
        for i in range(keypoints_2d_in_view.shape[0])
        if i != pelvis_i
    ])).unsqueeze(0)


def normalize_keypoints(keypoints_2d, pelvis_center_kps, normalization, pelvis_i):
    """ "divided by its Frobenius norm in the preprocessing" """

    batch_size, n_views = keypoints_2d.shape[0], keypoints_2d.shape[1]

    if pelvis_center_kps:
        kps = center2pelvis(keypoints_2d, pelvis_i)
    else:
        kps = keypoints_2d

    if normalization == 'd2pelvis':
        scaling = torch.cat([
            torch.max(
                torch.cat([
                    dist2pelvis(kps[batch_i, view_i])
                    for view_i in range(n_views)
                ]).unsqueeze(0)
            ).unsqueeze(0).repeat(1, n_views)  # same for each view
            for batch_i in range(batch_size)
        ])
    elif normalization == 'fro':
        scaling = torch.cat([
            torch.cat([
                torch.norm(kps[batch_i, view_i], p='fro').unsqueeze(0)
                for view_i in range(n_views)
            ]).unsqueeze(0)
            for batch_i in range(batch_size)
        ])
    elif normalization == 'maxfro':
        scaling = torch.cat([
            torch.max(
                torch.cat([
                    torch.norm(kps[batch_i, view_i], p='fro').unsqueeze(0)
                    for view_i in range(n_views)
                ]).unsqueeze(0)
            ).unsqueeze(0).repeat(1, n_views)  # same for each view
            for batch_i in range(batch_size)
        ])
    elif normalization == 'fixed':
        factor = 40.0  # todo to be scaled with K
        scaling = factor * torch.ones(batch_size, n_views)

    return torch.cat([
        torch.cat([
            (
                kps[batch_i, view_i] / scaling[batch_i, view_i]
            ).unsqueeze(0)
            for view_i in range(n_views)
        ]).unsqueeze(0)
        for batch_i in range(batch_size)
    ])
