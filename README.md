# Nowcasting: Deep Learning Framework for Precipitation Forecasting

This repository provides a flexible and extensible deep learning framework for **precipitation nowcasting**, implementing several models including **EarthFormer**, **LDCAST**, and **DGMR**. It supports multiple data sources such as radar, satellite, and weather station (AWS) data, and provides benchmarking tools for both deep learning and traditional methods.

---

## Overview

This framework enables training, evaluation, and benchmarking of cutting-edge models for short-term precipitation forecasting. Built with **PyTorch** and **PyTorch Lightning**, it includes a modular dataloader (based on Zarr) and standardized evaluation tools.

### Key Features

* Implementations of deep learning models for nowcasting
* Supports radar, satellite, ground station, and NWP data
* Modular configuration and data preprocessing
* Benchmarking utilities for model comparison
* Lightning-based training with built-in checkpointing and logging

---

## Repository Structure

```
config/         # YAML configuration files for models and experiments
data/           # Processed datasets and intermediate storage
models/         # Pretrained model weights (optional)
nowcasting/     # Core package
  ├── data/     # Data loaders and preprocessing
  ├── models/   # Model architectures (EarthFormer, LDCAST, etc.)
  ├── metrics/  # Evaluation metrics (CSI, MSE, etc.)
  └── benchmarks/ # Benchmarking scripts for all models
scripts/        # Training, evaluation, and plotting scripts
```

---

## Implemented Models

### EarthFormer

Transformer-based model designed for Earth system forecasting. Adapted from [Amazon's Earth Forecasting Transformer](https://github.com/amazon-science/earth-forecasting-transformer).

### LDCAST (Latent Diffusion Cast)

A diffusion-based model from [MeteoSwiss](https://github.com/MeteoSwiss/ldcast). Also supports training the deterministic component (referred to as "NowcastNet" in this repo—not to be confused with other models of the same name).

### DGMR-UK (Deep Generative Model of Rainfall)

Adapted from DeepMind’s official [DGMR implementation](https://github.com/google-deepmind/deepmind-research/tree/master/nowcasting). Trained on UK radar data.

### PySTEPS (Traditional Baseline)

Optical flow-based model using the [PySTEPS library](https://pysteps.github.io/), included for baseline comparisons with deep learning models.

---

## Installation

> **Python >= 3.11** is required due to dependency constraints.

It is recommended to install the framework in a virtual environment (e.g. `venv`, `conda`, or `pyenv`).

```bash
# Clone the repository
$ git clone https://github.com/<your-username>/nowcasting.git
$ cd nowcasting

# Install dependencies
$ pip install -e .
```

This installs all required dependencies and installs the project in "editable" mode.

---

## Training a Model

To start training:

```bash
$ python train.py --config="./config/<config_name>.yaml"
```

### Checkpoints and Logs

Training logs and checkpoints are stored in:

```
./results/tb_logs/<experiment_name>/version_<number>/
```

This includes:

* A copy of the config file
* TensorBoard logs (`events.out.tfevents`)
* Checkpoints (`checkpoints/`) for resuming training

To resume from a previous training run:

```bash
$ python scripts/train.py --version_path="./results/tb_logs/<experiment_name>/version_<number>"
```

---

## Evaluation

Evaluations are conducted using the `Evaluator` class from `nowcasting.metrics.validation`, which computes and stores various performance metrics (e.g., MSE, CSI, F1) based on model outputs.

### Evaluation with Trained Models (e.g. EarthFormer, LDCAST)

To evaluate a model **you’ve trained**, you must:

1. **Update the Config File**
   Edit the config file used for training (now located in your output directory) and set the correct checkpoint path:

   ```yaml
   model:
     checkpoint_path: "./results/tb_logs/<experiment_name>/version_<number>/checkpoints/<your_checkpoint>.ckpt"
   ```

   > ⚠️ If `checkpoint_path` is not set, the model will use random initialization, leading to poor evaluation results.

2. **Run the Evaluation**

   ```bash
   $ python scripts/eval.py --config="./results/tb_logs/<experiment_name>/version_<number>/<config_name>.yaml"
   ```

---

### Evaluation with Static/Baseline Models (e.g. DGMR-UK, PySTEPS)

These models do not require training and have fixed or pretrained weights. Use the corresponding config from the `config/` directory:

```bash
$ python scripts/eval.py --config="./config/<config_name>.yaml"
```

---

### Output Location

All evaluation results are saved to:

```
./results/scores/<eval_dir>/<eval_name>/
```

Both `eval_dir` and `eval_name` are specified in the config file used for evaluation.

---

## Plotting Results

To visualize evaluation scores:

1. Open `scripts/plot_scores.py`
2. Set the `models` (i.e. `eval_name`) and `dir` (`eval_dir`) values in the script
3. Ensure all models use the same lead times and thresholds
4. Run:

```bash
$ python scripts/plot_scores.py
```

---

## Data Preparation

The framework supports:

* Grid projection alignment
* Rain rate normalization
* Z-score normalization
* Interpolation for AWS data
* Resizing with multiple interpolation methods

> **Note:** Most data is preprocessed to a common grid. Only variable resolutions may differ across sources.

---

## Configuration

Configuration files (`.yaml`) are located in `config/`. Some examples:

* `earthformer.yaml` – EarthFormer base setup
* `ldcast_nowcast.yaml` – LDCAST configuration
* `dgmr_uk.yaml` – DGMR (UK radar) setup
* `pysteps.yaml` – Optical flow baseline

For more details, see [`config/README.md`](config/README.md).

---

## Questions / Contributions

Feel free to open issues or pull requests! If you want help extending the framework or adding new models, we welcome contributions and suggestions.

---

## TODO

Dataset:
* Fix: add `"downscale_factor": 1` to `dataset.zarr/radar/zarr.json`
* Fix: add new array to AWS that is the union over all the 'timemasks'
* Update the metadata to be compilenet with the  [NetCDF convention](https://www.seadatanet.org/Standards/Data-Transport-Formats)
* Make dataset compatable with xarray
Models:
* Add custom DGMR pytoch implimentation to the reposotory
* Create seperation layer between the trainer and the models. This layer will need to deliver a Lighting model and will have to be implimented for each of the models. This will make sure that we do not have to modify the original model code too much.
Evaluation:
* Add more plotting fucntionaly
* Add functionaly for saving some/all of the evaluation model output
* Increase preformance by having also implementing a pytorch version, so that the tensors do not have to be converted so much.

---
