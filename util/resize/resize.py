import os
import argparse
import json

import cv2


def resize(path: str, size: tuple[int, int], image_suffix, depth_suffix, normal_suffix, extension):
    """Resizes the images in the dataset to the given size and generates a new JSON file.

    Args:
        path (str): The path to the dataset JSON file.
        size (tuple[int, int]): The size to resize the images to.
        image_suffix (str, optional): The suffix of the image files.
            Defaults to ''.
        depth_suffix (str, optional): The suffix of the depth files.
            Defaults to '_depth_0001.png'.
        normal_suffix (str, optional): The suffix of the normal files.
            Defaults to '_normal_0001.png'.
        extension (str, optional): The extension of the files (e.g. '.png')
            Defaults to '.png'.
    """
    # Read the JSON file and get the camera angle and frames.
    with open(path, 'r') as f:
        data = json.load(f)
    frames = data['frames']
    images_directory = os.path.join(
        # path to directory of JSON file
        os.path.abspath(os.path.join(path, os.pardir)),
        os.path.dirname(os.path.relpath(
            frames[0]['file_path'])) + f'_{size[0]}x{size[1]}')  # directory of resized images
    os.makedirs(images_directory, exist_ok=True)
    image_suffix = image_suffix if image_suffix is not None else ''
    image_suffix = image_suffix + extension
    depth_suffix = depth_suffix + extension
    normal_suffix = normal_suffix + extension
    # Resize the images.
    for frame in frames:
        image_path = os.path.join(
            os.path.dirname(path),
            os.path.relpath(frame['file_path']))
        new_path = os.path.join(images_directory, os.path.basename(
            frame['file_path']))
        image = cv2.imread(image_path + image_suffix, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, size)
        cv2.imwrite(new_path + extension, image)
        if os.path.exists(image_path + depth_suffix):
            depth = cv2.imread(
                image_path + depth_suffix, cv2.IMREAD_UNCHANGED)
            depth = cv2.resize(depth, size)
            cv2.imwrite(new_path + depth_suffix, depth)
        if os.path.exists(image_path + normal_suffix):
            normal = cv2.imread(
                image_path + normal_suffix, cv2.IMREAD_UNCHANGED)
            normal = cv2.resize(normal, size)
            cv2.imwrite(new_path + normal_suffix, normal)
        frame['file_path'] = './' + \
            os.path.relpath(new_path, os.path.dirname(path))
    # Write the new JSON file. Name is same as original JSON file but with the size appended.
    with open(os.path.join(os.path.dirname(path), os.path.basename(
            path).split('.')[0] + f'_{size[0]}x{size[1]}.json'), 'w') as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path to dataset JSON file')
    parser.add_argument('width', type=int, help='width to resize images to')
    parser.add_argument('height', type=int, help='height to resize images to')
    parser.add_argument('--image-suffix', type=str, default=None,
                        help='suffix of images')
    parser.add_argument('--depth-suffix', type=str, default='_depth_0001',
                        help='suffix of depth maps')
    parser.add_argument('--normal-suffix', type=str, default='_normal_0001',
                        help='suffix of normal maps')
    parser.add_argument('--extension', type=str, default='.png',
                        help='file extension of images')
    args = parser.parse_args()
    resize(args.path, (args.width, args.height),
           args.image_suffix, args.depth_suffix, args.normal_suffix, args.extension)


if __name__ == '__main__':
    main()
