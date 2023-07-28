import torch
from torch import Tensor


def get_rays(
    image_height: int, image_width: int, camera_angle_x: float, pose_matrix: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns a tensor of rays for the given image size, camera angle, and pose matrix.

    Args:
        image_height (int): The height of the image in pixels.
        image_width (int): The width of the image in pixels.
        camera_angle_x (float): The horizontal field of view of the camera in radians.
        pose_matrix (Tensor): The pose matrix of the camera.

    Returns:
        tuple[Tensor, Tensor]: Two Tensors each of shape (image_height, image_width, 3),
            where the first Tensor is the origins of the rays and the second Tensor is
            the directions of the rays.
    """
    # Calculate view plane width assuming view plane is at z = -1 in camera space
    view_plane_width = 2 * torch.tan(torch.tensor(camera_angle_x) / 2)
    # Calculate view plane height from view plane width and image aspect ratio
    view_plane_height = view_plane_width * image_height / image_width

    # Create tensors of pixel indices
    # j is [0, 1, ..., image_width - 1]
    # i is [0, 1, ..., image_height - 1]
    j, i = torch.meshgrid(
        torch.arange(image_width), torch.arange(image_height), indexing="xy"
    )
    j = j.to(pose_matrix.device)
    i = i.to(pose_matrix.device)

    # Calculate x and y coordinates of the view plane
    x = (
        j + 0.5
    ) / image_width * view_plane_width - view_plane_width / 2  # [image_height, image_width]
    y = (
        image_height - 1 - i + 0.5
    ) / image_height * view_plane_height - view_plane_height / 2  # [image_height, image_width]
    # View plane is at z = -1 in camera space
    z = -torch.ones_like(x)  # [image_height, image_width]

    # Non-normalized camera space directions
    # Length of direction vector is proportional to corresponding view plane position's distance from the camera
    # [image_height, image_width, 3]
    directions = torch.stack([x, y, z], dim=-1)

    # Upper left 3x3 submatrix of pose matrix is the rotation matrix
    rotation_matrix = pose_matrix[:3, :3]
    # Upper right column of pose matrix is the translation vector
    translation = pose_matrix[:3, 3]

    # Rotate direction vectors
    directions = torch.matmul(
        rotation_matrix[None, None, ...], directions[..., None]
    ).squeeze(-1)

    # Normalize direction vectors
    norms = torch.norm(directions, dim=-1, keepdim=True)
    directions_normalized = directions / norms

    # Translation is broadcasted to [image_height, image_width, 3]
    origins = translation[None, None, ...].expand_as(directions)

    # [image_height, image_width, 3]
    return origins, directions, directions_normalized
