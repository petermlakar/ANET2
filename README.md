# AtmosphereNET 2.0: Neural network based probabilistic post-processing of ensemble weather forecasts

<p align="center">
    <img src="images/logo_anet2.png" alt="ANET2 logo" width="300px">
</p>

AtmosphereNET 2.0 (ANET2) is a state-of-the-art neural network algorithm for probabilistic ensemble weather forecast post-processing using normalizing flows as parametric distribution models.
ANET2 is able to model varying distributions without the need to specify the target distribution beforehand.

## Requirements

To train the ANET2 model the following Python libraries are required and can be installed with pip:
```console
pip3 install numpy==1.23.5 torch=1.13.1 netCDF4==1.6.2 xarray==2022.11.0
```

## Usage

### Training the model

The training script facilitates the training of three different ANET2 variants, described in more detail in Mlakar et al., [2023](https://doi.org/10.48550/arXiv.2303.17610).
To train any of the three ANET2 variants:

```console
python3 train.py ANET2 /path/to/data/folder

python3 train.py ANET2_BERN /path/to/data/folder

python3 train.py ANET2_NORM /path/to/data/folder
```



### Inference

## Example

## Publication
