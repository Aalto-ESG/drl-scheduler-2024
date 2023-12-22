# DRL-scheduler for conditional task graphs

Python prototype for a DRL-based scheduler.

This repository acts as a reference implementation for our paper. It includes both the simulator and the 
training code for the DRL-agent as described in the paper.

## Installation

Ray can be very finicky when running on Windows. It is recommended to use linux or WSL. Otherwise, you may need
to iterate ray versions until you find something that works.

Recommended way to install requirements is to use Conda (Miniconda or Anaconda).

Create new conda env with:

``` conda env create --no-default-packages -f environment-ray-nocuda.yaml ```

or update an existing conda env:

``` conda env update --file environment-ray-nocuda.yaml --prune ```

## Usage

#### 1. Train and evaluate policies

Running with default settings:

``python experiment_runner``

Depending on configuration, this will produce model checkpoints and evaluation results. 

Processed results are also saved as csv files in ``plotting/csv`` for easy plotting.

#### 2. Plot results

Jupyter notebooks included in ``plotting`` folder can be used to create plots from the evaluation results.



## Configuration

The project uses Hydra for managing configurations. Most of the code supports overriding individual parameters
by giving them as arguments:

``python experiment_runner sim.interval=5``

However, not everything is included in the configuration files. Some of the parameters, such as training hyperparameters,
are hardcoded in the training code. Some of the files may even include hardcoded overrides for the Hydra config.

Hardcoded configs are especially included in:

- ``tile_simulator.py``
- ``ray_train_pbt.py``
- ``experiment_runner.py``

## TODO

- Clean up code
  - There is a lot of leftover code from previous experiments
  - There are some hard coded overrides for configs in various locations
  - There are some leftover variables in configuration files that do nothing
  - Evaluation code has gotten unnecessarily convoluted
- Improve documentation

