# EZNeRF: Neural Radiance Fields Explained!

## What is a NeRF?

In simplest terms, a neural radiance field (NeRF) represents a 3D scene as a neural network which can then be queried to _synthesize novel views_ of the scene. In other words, given a set of images of a scene captured from different angles, a NeRF learns to represent the scene and can produce images of the scene from any angle. This is how we generate the videos of the scenes above from a limited set of images!

More specifically, NeRF learns to predict the color and volume density corresponding to any point and space and a viewing direction. By then accumulating color and volume density for samples along rays in space, we can volume render novel views of the scene.

> NeRF was first introduced by Mildenhall et. al. in the seminal paper [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934). The original code for the paper can be found [here](https://github.com/bmild/nerf).

## What is EZNeRF?

EZNeRF is an implementation of NeRF largely following the original paper designed with the goal of having _organized, documented, and easy-to-understand code._

Markdown explanations corresponding to each code file are provided, and they explain both higher-level intuition as well as per-line intention for each part of the implementation.

NeRF is not only promising in the fields of computer vision and computer graphics with important applications such as in AR/VR, but it is also a fantastic opportunity to learn about and integrate many concepts in computer graphics, computer vision, deep learning, and statistics.

The goal of EZNeRF is to thoroughly communicate all of these ideas so that a reader (who may be a student, researcher, engineer, or any curious individual) may understand how all of these concepts tie together and produce the results you see above.

EZNeRF does not attempt to be the most efficient or performant implementation of NeRF, nor does it attempt to be the most complete. In fact, EZNeRF currently only supports synthetic scenes with provided camera poses. If you are looking for implementations that are suited for real applications, there are many other implementations and variants of NeRF that produce better results _much_ faster. Check out [Nerfstudio](https://nerf.studio/) and [InstantNGP](https://github.com/NVlabs/instant-ngp) for some great examples.

## Getting Started

EZNeRF is implemented in Python with PyTorch and a couple of other libraries. This project uses [`uv`](https://github.com/astral-sh/uv), a fast Python package and project manager. It has a similar interface to `pip`, but it also helps create virtual environments (like `conda`) and has reproducible dependency management with a [lockfile](uv.lock).

### Installing Dependencies

After cloning the repository, you can install the required dependencies by navigating to the root directory of the repository and running:

```sh
uv sync --only-managed
```

This will have `uv` create a virtual environment at `.venv` and install all of the dependencies specified in [`pyproject.toml`](pyproject.toml).

> [!NOTE]
> You might be used to activating virtual environments (such as with `conda activate` or `source .venv/bin/activate`). You won't need to activate the virtual environment with `uv`, as we will use `uv run` to run our scripts.

### Downloading Data

We use the same data as the original implementation.
From the root directory of the repository, run:

```sh
./util/download_example_data.sh
```

There should now be a `data` directory at the root of the repository.

#### Resizing the data

You will likely need to resize the data for faster training or reduced memory usage, as the original synthetic images are each 800x800.

We provide a `resize.py` script which will resize the images to a given size. Details on the script's usage are provided in the [`resize.py` README](util/resize/README.md).

## Running EZNeRF

Detailed documentation on the training and testing scripts can be found in the [`train.py` README](train/README.md) and [`test.py` README](test/README.md), respectively.

### Pre-trained Weights

To be made available soon.

### `eznerf.py`

In most cases, all you will probably need to run is the combined `eznerf.py` script. You can see the full list of arguments, their descriptions, and default values by running:

```sh
uv run eznerf.py -- --help
```

#### Example Usage

To train a model on the example data with 400x400 images, run:

```sh
uv run eznerf.py --h 400 --w 400
```

### WandB

EZNeRF uses [WandB](https://wandb.ai/) to log training metrics and visualizations. When you begin a training run, you should see a link to the run's dashboard printed to the terminal. Clicking on the link will allow you to view your training metrics and validation outputs.

The code allows for anonymous logging by default, but you can specify your own WandB API key if you'd like to store your results on your account or keep them private:

```sh
uv run wandb login
```
