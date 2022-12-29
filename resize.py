import json
import os
import cv2
from data import PathFrame
import argparse


def resize(path: str, size: tuple[int, int]):
    """Resizes the images in the dataset to the given size and generates a new JSON file.

    Args:
        path (str): The path to the dataset JSON file.
        size (tuple[int, int]): The size to resize the images to.
    """
    # Read the JSON file and get the camera angle and frames.
    with open(path, 'r') as f:
        data = json.load(f)
    frames: list[PathFrame] = data['frames']
    images_directory = os.path.join(
        # path to directory of JSON file
        os.path.abspath(os.path.join(path, os.pardir)),
        os.path.dirname(os.path.relpath(
            frames[0]['file_path'])) + f'_{size[0]}x{size[1]}')  # directory of resized images
    os.makedirs(images_directory, exist_ok=True)
    # Resize the images.
    for frame in frames:
        image_path = os.path.join(
            os.path.dirname(path),
            os.path.relpath(frame['file_path'] + '.png'))
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, size)
        new_path = os.path.join(images_directory, os.path.basename(
            frame['file_path']) + '.png')
        cv2.imwrite(new_path, image)
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
    args = parser.parse_args()
    resize(args.path, (args.width, args.height))


if __name__ == '__main__':
    main()
