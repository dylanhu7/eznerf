# Resize
The `resize.py` script currently only supports resizing the synthetic data with the same format as that in `data/nerf_synthetic/lego`.

Given the path to the transforms JSON file (e.g. `data/nerf_synthetic/lego/transforms_train.json`), the script will resize the images and save them to a new directory, as well as generate a new corresponding JSON file. This new JSON file can then be used to train/test the model with `eznerf.py`.

## Arguments
| Argument | Description | Default |
| --- | --- | --- |
| `path` | First positional argument. Path to the transforms JSON file. Required. | `None` |
| `width` | Width of the resized images. Required. | `None` |
| `height` | Height of the resized images. Required. | `None` |
| `--image-suffix` | A suffix to append to the image filenames. | `None` |
| `--depth-suffix` | A suffix to append to the depth image filenames. | `_depth_0001` |
| `--normal-suffix` | A suffix to append to the normal image filenames. | `_normal_0001` |
| `--extension` | The file extension for the produced images. | `.png` |

## Example Usage
To resize the given LEGO data to 80x80, run the following command from the project root directory:
```sh
python util/resize/resize.py data/nerf_synthetic/lego/transforms_train.json 80 80
```
A new transforms JSON file will be created as `data/nerf_synthetic/lego/transforms_train_80x80.json`, and the resized images will be saved to `data/nerf_synthetic/lego/train_80x80/`.