# [WIP] EZNeRF: Neural Radiance Fields Explained!

## What is a NeRF?
In simplest terms, a neural radiance field (NeRF) represents a 3D scene as a neural network which can then be queried to *synthesize novel views* of the scene. In other words, given a set of images of a scene captured from different angles, a NeRF learns to represent the scene and can produce images of the scene from any angle. This is how we generate the videos of the scenes above from a limited set of images!

More specifically, NeRF learns to predict the color and volume density corresponding to any point and space and a viewing direction. By then accumulating color and volume density for samples along rays in space, we can volume render novel views of the scene.

> NeRF was first introduced by Mildenhall et. al. in the seminal paper [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934). The original code for the paper can be found [here](https://github.com/bmild/nerf).

## What is EZNeRF?
EZNeRF is an implementation of NeRF largely following the original paper designed with the goal of having *organized, documented, and easy-to-understand code.*

Markdown explanations corresponding to each code file are provided, and they explain both higher-level intuition as well as per-line intention for each part of the implementation.

NeRF is not only promising in the fields of computer vision and computer graphics with important applications such as in AR/VR, but it is also a fantastic opportunity to learn about and integrate many concepts in computer graphics, computer vision, deep learning, and statistics.

The goal of EZNeRF is to thoroughly communicate all of these ideas so that a reader (who may be a student, researcher, engineer, or any curious individual) may understand how all of these concepts tie together and produce the results you see above.

EZNeRF does not attempt to be the most efficient or performant implementation of NeRF, nor does it attempt to be the most complete. In fact, EZNeRF currently only supports synthetic scenes with provided camera poses. If you are looking for implementations that are suited for real applications, there are many other implementations and variants of NeRF that produce better results *much* faster. Check out [Nerfstudio](https://nerf.studio/) and [InstantNGP](https://github.com/NVlabs/instant-ngp) for some great examples.

## Getting Started
EZNeRF is implemented in Python with PyTorch and a couple of other libraries. We recommend using Conda for managing dependencies in a virtual environment. [Miniforge](https://github.com/conda-forge/miniforge) is recommended, but any Conda installation should work. Alternatively, you can install the dependencies manually with `pip`.

### Installing Dependencies

After cloning the repository, you can install the required dependencies by navigating to the root directory of the repository and running:
```sh
conda env create -f environment.yaml
```
> This `environment.yaml` file includes `pytorch-cuda` so that machines with NVIDIA GPUs may leverage those resources. However, if this package fails to be found or install for your system (if you are using a Mac, for example) you can remove it from the `environment.yaml` file.

After creating the environment, activate it by running:
```sh
conda activate eznerf
```

### Downloading Data
We use the same data as the original implementation.
From the root directory of the repository, run:
```sh
./util/download_example_data.sh
```
There should now be a `data` directory at the root of the repository.

## Running EZNeRF

### `eznerf.py`
In most cases, all you will probably need to run is the combined [`eznerf.py`](./eznerf.py) script, which can take a variety of arguments. However, config files are supported, and we provide a basic config file for training the example LEGO scene in `config/lego.yaml`.

#### Example Usage
To train a model on the example data, run:
```sh
python eznerf.py --config config/lego.yaml
```

### WandB
Once you initiate a training run, the logging tool [WandB](https://wandb.ai/site) will provide you with a link to a dashboard where you can monitor the training progress and view the results of the model. WandB may prompt you to create an account if you do not already have one, but you can proceed anonymously if you wish.

### Pre-trained Weights
To be made available soon.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or would like to contribute.