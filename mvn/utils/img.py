import numpy as np
import cv2
from PIL import Image

import torch

IMAGENET_MEAN, IMAGENET_STD = np.array(
    [0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def crop_image(image, bbox):
    """Crops area from image specified as bbox. Always returns area of size as bbox filling missing parts with zeros
    Args:
        image numpy array of shape (height, width, 3): input image
        bbox tuple of size 4: input bbox (left, upper, right, lower)

    Returns:
        cropped_image numpy array of shape (height, width, 3): resulting cropped image

    """

    image_pil = Image.fromarray(image)
    image_pil = image_pil.crop(bbox)

    return np.asarray(image_pil)


def resize_image(image, shape):
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def get_square_bbox(bbox):
    """Makes square bbox from any bbox by stretching of minimal length side

    Args:
        bbox tuple of size 4: input bbox (left, upper, right, lower)

    Returns:
        bbox: tuple of size 4:  resulting square bbox (left, upper, right, lower)
    """

    left, upper, right, lower = bbox
    width, height = right - left, lower - upper

    if width > height:
        y_center = (upper + lower) // 2
        upper = y_center - width // 2
        lower = upper + width
    else:
        x_center = (left + right) // 2
        left = x_center - height // 2
        right = left + height

    return left, upper, right, lower


def scale_bbox(bbox, scale):
    left, upper, right, lower = bbox
    width, height = right - left, lower - upper

    x_center, y_center = (right + left) // 2, (lower + upper) // 2
    new_width, new_height = int(scale * width), int(scale * height)

    new_left = x_center - new_width // 2
    new_right = new_left + new_width

    new_upper = y_center - new_height // 2
    new_lower = new_upper + new_height

    return new_left, new_upper, new_right, new_lower


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def image_batch_to_numpy(image_batch):
    image_batch = to_numpy(image_batch)
    image_batch = np.transpose(image_batch, (0, 2, 3, 1))  # BxCxHxW -> BxHxWxC
    return image_batch


def image_batch_to_torch(image_batch):
    image_batch = np.transpose(image_batch, (0, 3, 1, 2))  # BxHxWxC -> BxCxHxW
    image_batch = to_torch(image_batch).float()
    return image_batch


def normalize_image(image):
    """Normalizes image using ImageNet mean and std

    Args:
        image numpy array of shape (h, w, 3): image

    Returns normalized_image numpy array of shape (h, w, 3): normalized image
    """
    return (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def denormalize_image(image):
    """Reverse to normalize_image() function"""
    return np.clip(255.0 * (image * IMAGENET_STD + IMAGENET_MEAN), 0, 255)


# todo faster
def resample_channel(image, target_intrinsics, intrinsics, padding_size=10):
    y_ = np.arange(0, image.shape[0])[:, np.newaxis].repeat(
        image.shape[1], 1
    ).reshape(-1, 1) + 0.5
    x_ = np.arange(0, image.shape[1])[:, np.newaxis].repeat(
        image.shape[0], 1
    ).T.reshape(-1, 1) + 0.5
    uv1 = np.concatenate((x_, y_, np.ones_like(x_)), 1)

    # map them to original image
    uv1_orig = (intrinsics @ np.linalg.inv(target_intrinsics) @ uv1.T).T

    # grid centers mapped to original image
    x = np.clip(
        uv1_orig[:, 0].astype(int),
        0 - padding_size,
        padding_size + image.shape[1] - 1
    )
    y = np.clip(
        uv1_orig[:, 1].astype(int),
        0 - padding_size,
        padding_size + image.shape[0] - 1
    )

    # pad image and copy values
    padded_image = np.zeros(
        (
            2 * padding_size + image.shape[0],
            2 * padding_size + image.shape[1]
        )
    ) * 255
    padded_image[padding_size:-padding_size,
                 padding_size:-padding_size] = image

    q = padded_image[padding_size + y, padding_size + x]

    resampled_img = q.reshape(image.shape[0], image.shape[1])
    resampled_img = resampled_img[
        0:2 * (target_intrinsics[0, 2].astype(int) + 1),
        0:2 * (target_intrinsics[1, 2].astype(int) + 1)
    ]

    return resampled_img


# thanks to @edo
def resample_image(image, target_intrinsics, intrinsics):
    r_img = resample_channel(image[:, :, 0], target_intrinsics, intrinsics)
    g_img = resample_channel(image[:, :, 1], target_intrinsics, intrinsics)
    b_img = resample_channel(image[:, :, 2], target_intrinsics, intrinsics)

    return np.concatenate(
        (
            r_img[:, :, np.newaxis],
            g_img[:, :, np.newaxis],
            b_img[:, :, np.newaxis],
        ),
        2
    )


def make_with_target_intrinsics(image, intrinsics, target_intrinsics):
    height, width = image.shape[:2]

    # compute new shape
    f_x = intrinsics[0, 0]
    f_x1 = target_intrinsics[0, 0]
    scale_w = f_x1 / f_x
    new_w = width * scale_w

    f_y = intrinsics[1, 1]
    f_y1 = target_intrinsics[1, 1]
    scale_h = f_y1 / f_y
    new_h = height * scale_h

    # compute cropping after scaling
    x_new = intrinsics[0, 2] * scale_w
    crop_left = x_new - target_intrinsics[0, 2]

    y_new = intrinsics[1, 2] * scale_h
    crop_upper = y_new - target_intrinsics[1, 2]

    scaling = (new_w, new_h)
    cropping = (
        crop_left,
        crop_upper,
        width * scale_w,  # no cropping right
        height * scale_h  # no cropping bottom
    )

    return scaling, cropping


def rotation_matrix_from_vectors_rodrigues(vec1, vec2):
    """ https://stackoverflow.com/a/59204638/7643222 based on https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula"""

    a, b = (
        (vec1 / np.linalg.norm(vec1)).reshape(3),  # normalize
        (vec2 / np.linalg.norm(vec2)).reshape(3)
    )

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))  # 3 x 3


def rotation_matrix_from_vectors_kabsch(vec1, vec2):
    """ https://github.com/scipy/scipy/blob/master/scipy/spatial/transform/rotation.pyx#L2204 """

    from scipy.spatial.transform import Rotation as R

    return R.align_vectors(
        np.expand_dims(vec1, axis=0),
        np.expand_dims(vec2, axis=0)
    )[0].as_matrix()


def rotation_matrix_from_vectors_torch(vec1, vec2):
    """ see `rotation_matrix_from_vectors` """

    dev = vec1.device
    a, b = (
        (vec1 / torch.norm(vec1)).double(),
        (vec2 / torch.norm(vec2)).double()
    )

    v = torch.cross(a, b)
    c = torch.dot(a, b)
    s = torch.norm(v)
    kmat = torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], requires_grad=True).to(dev).double()

    return torch.eye(3).to(dev) +\
        kmat +\
        torch.mm(kmat, kmat) * ((1 - c) / (torch.square(s)))  # 3 x 3
