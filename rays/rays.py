import torch


def get_rays(image_width: int, image_height: int, camera_angle_x: float, pose_matrix: torch.Tensor) -> torch.Tensor:
    """Returns a tensor of rays for the given image size, camera angle, and pose matrix.

    Args:
        image_width (int): The width of the image in pixels.
        image_height (int): The height of the image in pixels.
        camera_angle_x (float): The horizontal field of view of the camera in radians.
        pose_matrix (torch.Tensor): The pose matrix of the camera.

    Returns:
        torch.Tensor: A tensor of rays of shape (image_height, image_width, 3, 2), where the last dimension partitions the origins and directions.
    """
    # Calculate view plane width assuming view plane is at z = -1 in camera space
    view_plane_width = 2 * torch.tan(torch.tensor(camera_angle_x) / 2)
    # Calculate view plane height from view plane width and image aspect ratio
    view_plane_height = view_plane_width * image_height / image_width

    # Create tensors of pixel indices
    # j is [0, 1, ..., image_width - 1]
    # i is [0, 1, ..., image_height - 1]
    j, i = torch.meshgrid(torch.arange(image_width),
                          torch.arange(image_height),
                          indexing='xy')
    j = j.to(pose_matrix.device)
    i = i.to(pose_matrix.device)

    # Calculate x and y coordinates of the view plane
    x = (j + 0.5) / image_width * view_plane_width - \
        view_plane_width / 2  # [image_height, image_width]
    y = (image_height - 1 - i + 0.5) / image_height * \
        view_plane_height - view_plane_height / \
        2  # [image_height, image_width]
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
    directions = torch.matmul(directions, rotation_matrix)

    # Translation is broadcasted to [image_height, image_width, 3]
    origins = translation[None, None, ...].expand_as(directions)

    # [image_height, image_width, 3, 2]
    return torch.stack([origins, directions], dim=-1)
