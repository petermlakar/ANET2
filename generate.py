import torch
import torch.jit as jit
import numpy as np

from os.path import exists, join
from os import mkdir, listdir

import sys
from dataset import Databank, Dataset, norm, standardize, destandardize, load_training_dataset, load_test_dataset 
import netCDF4

from time import time as timef

import json

CUDA = False #torch.cuda.is_available() 

#########################################################

CFG_PATH = "" if len(sys.argv) == 1 else sys.argv[1]

cfg = json.load(open(join(CFG_PATH, "config.json")))

OUTPUT_PATH = cfg["generate"]["outputPath"] 
MODELS_PATH = cfg["generate"]["modelsPath"]
BEST_MODEL_ONLY    = cfg["generate"]["useBestModelOnly"]
AVERAGE_PARAMETERS = cfg["generate"]["averageParameters"]
RESIDUALS = cfg["generate"]["residuals"]

DATA_PATH = cfg["dataPath"]

if not exists(OUTPUT_PATH):
    mkdir(OUTPUT_PATH)

print("Generating predictions with model:", MODELS_PATH.split("/")[-1], " -> using residuals: ", RESIDUALS)

#########################################################
# Load training data, remove nan 

_, _,\
_, _,\
_,\
X_mean, X_std,\
Y_mean, Y_std,\
P_mean, P_std,\
_, _,\
_, _ = load_training_dataset(DATA_PATH, residuals = RESIDUALS)

X, Y, P,\
time, stations,\
valid_time = load_test_dataset(DATA_PATH)

print(X.shape)

X = standardize(X, X_mean, X_std)
Y = standardize(Y, Y_mean, Y_std)
P = standardize(P, P_mean, P_std)

#########################################################

BATCH_SIZE = 512

bank = Databank(X, Y, P, valid_time, cuda = CUDA)
dataset = Dataset(bank, bank.index, batch_size = BATCH_SIZE, train = False, cuda = CUDA)

#########################################################

models_losses = []
models_regression  = []
model_distribution = None

for m in listdir(MODELS_PATH):

    models_regression.append(torch.jit.load(join(MODELS_PATH, m, "model_regression"), map_location = torch.device("cpu")))
    models_losses.append(np.loadtxt(join(MODELS_PATH, m, "Valid_loss")).min())
    model_distribution = model_distribution if model_distribution is not None else torch.jit.load(join(MODELS_PATH, m, "model_distribution"), map_location = torch.device("cpu"))

    if CUDA:
        models_regression[-1] = models_regression[-1].to("cuda:0")

    models_regression[-1].eval()

if CUDA:
    model_distribution = model_distribution.to("cuda:0")

print(model_distribution.nblocks, model_distribution.nknots)

if BEST_MODEL_ONLY:

    i = np.where(np.array(models_losses) == np.array(models_losses).min())[0][0]

    models_regression = [models_regression[i]]
    models_losses = models_losses[i]

    print(f"Best only mode: loss -> {models_losses}")

#########################################################

nbins = 51

P = np.zeros(Y.shape + (nbins,), dtype = np.float32)
ENS_C = np.zeros(X.shape, dtype = np.float32)

q = torch.linspace(0.01, 0.99, nbins, dtype = torch.float32)
q = torch.reshape(q, (1, 1, q.shape[0]))
q = q.cuda() if CUDA else q

print(f"Number of quantiles: {q.shape} |  number of required parameters: 21x{model_distribution.number_of_outputs//21}")

#########################################################

with torch.no_grad():
    
    start_time = timef()

    for i in range(len(dataset)):

        if (i + 1) % 100 == 0:
            print(f"Inference: {i}/{len(dataset)}")

        x, p, y, j = dataset[i]
        qtmp = q.expand(y.shape[0], y.shape[1], -1)

        if AVERAGE_PARAMETERS:

            parameters = []

            for model in models_regression:
                parameters.append(model(x, p))

            parameters = torch.stack(parameters, dim = 0).mean(dim = 0)
            model_distribution.set_parameters(parameters)

            f = model_distribution.iF(qtmp)

        else:

            f = []

            for model in models_regression:

                model_distribution.set_parameters(model(x, p))
                f.append(model_distribution.iF(qtmp))

            f = torch.stack(f, dim = 0).mean(dim = 0)

        if RESIDUALS:
            P[j[0, :], j[1, :], :, :] = (f*Y_std + (x*X_std + X_mean).mean(axis = -1)[..., None]).detach().cpu().numpy()
        else:
            P[j[0], j[1], :, :]  = f.detach().cpu().numpy()*X_std + X_mean

            print( P[j[0], j[1], :, :].min(), P[j[0], j[1], :, :].max(), x.mean())

    print(f"Execution time time: {timef() - start_time} seconds")

np.save(join(OUTPUT_PATH, "forecasts"), P)

