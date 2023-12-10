# AtmosphereNET 2.0: Neural network based probabilistic post-processing of ensemble weather forecasts

<p align="center">
    <img src="images/logo_anet2.png" alt="ANET2 logo" width="300px">
</p>

**AtmosphereNET 2.0** (**ANET2**) is a state-of-the-art neural network algorithm for probabilistic ensemble weather forecast post-processing.
It can formulate forcecast distributions based on three different techniques of choice: 
* normal distribution (**ANET2 Norm**)
* Bernstein quantile regression (**ANET2 Bern**)
* Normalizing flows (**ANET2 Flow**)
using normalizing flows as parametric distribution models.
**ANET2** is able to model varying distributions without the need to specify the target distribution beforehand.

## Requirements

To train the ANET2 model the following Python libraries are required and can be installed with pip:
```console
pip3 install numpy==1.23.5 torch=1.13.1 netCDF4==1.6.2 xarray==2022.11.0
```

## Usage

### Training the model

The training script facilitates the training of three different **ANET2** variants, described in more detail in Mlakar et al., [2023](https://doi.org/10.48550/arXiv.2303.17610).
The three **ANET2** variants are:
* **Flow**: The **ANET2** variant using normalizing flows as parametric distribution models in conjunction with the **ANET2** parameter estimation network
```console
python3 train.py FLOW /path/to/EUPPBench/data/folder
```
* **Bern**: Bernstein quantile regression used as the parametric distribution model in conjunction with the **ANET2** parameter estimation network
```console
python3 train.py BERN /path/to/EUPPBench/data/folder 
```
* **Norm**: Normal distribution used as the parametric distribution model in conjunction with the **ANET2** parameter estimation network
```console
python3 train.py NORM /path/to/EUPPBench/data/folder
```

### Train model on custom data

**ANET2** and its variants can also be trained on custom data with a varied amount of per-station predictors and custom lead time.
```python

import torch
from torch.optim import Adam

from models.ANET2 import Model

LEARNING_RATE = 1e-3
LEAD_TIME = 21                      # Lead time can be modified according to specific dataset needs
NUMBER_OF_PER_STATION_PREDICTORS = 9 # Number of per-station predictors can be modified according to specific dataset needs


model = Model(number_of_predictors = NUMBER_OF_PER_STATION_PREDICTORS, lead_time = LEAD_TIME)

if torch.cuda.is_available(): model = model.cuda()
model.train()

opt = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = 1e-6)


####
# Prepare training dataset
# Each batch returned from the training dataset should have the following format:
# 
# Training batch B := (x, p, y)
#
# x shape: [batch size, lead time, number of ensemble members] -> ensemble forecasts
# p shape: [batch size, number of per-station predictors]      -> per-station predictors
# y shape: [batch size, lead time]                             -> observations
#

D = ...
####

# Conduct training for one epoch (one pass through the training dataset)

for i in range(len(D)):
	
	x, p, y = D.__getitem__(i)

	loss = model.loss(x, p, y)

	opt.zero_grad()
        loss.backward()
        opt.step()

```


### Inference

If the train.py script is used for training it outputs the following folder
```console
trained model folder := <MODEL_NAME>_<LEARNING_RATE>_<BATCH_SIZE>_<TIMESTAMP><POSTFIX>
```
which contains the trained **ANET2** model files.
The folder can then be passed to the generate.py script to construct the inferences for the EUPPBench test dataset.
These will be saved at a custom, used defined location with the **_inference_output** postfix.

```console
python3 generate.py /inference/output/path /trained/model/folder/path /path/to/EUPPBench/data/folder 
```

## Publication

Peter Mlakar, Janko Merše, and Jana Faganeli Pucer. (2023). Ensemble weather forecast post-processing with a flexible probabilistic neural network approach.

Cite as:

```console
@misc{mlakar2023ensemble,
      title={Ensemble weather forecast post-processing with a flexible probabilistic neural network approach}, 
      author={Peter Mlakar and Janko Merše and Jana Faganeli Pucer},
      year={2023},
      eprint={2303.17610},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
