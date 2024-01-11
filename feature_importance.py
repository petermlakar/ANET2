import torch
import torch.jit as jit
import numpy as np

from os.path import exists, join
from os import mkdir, listdir

import sys
from dataset import Databank, Dataset, norm, normalize, denormalize, load_training_dataset, load_test_dataset 
import netCDF4

from time import time as timef

import json

def crps(x, y):

        shp = y.shape
    
        y = y[..., None]

        p = np.arange(1.0/x.shape[-1], 1.0, step = 1.0/x.shape[-1])[None]

        x = np.reshape(x, (np.prod(shp), x.shape[-1]))
        y = np.reshape(y, (np.prod(shp), y.shape[-1]))

        bin = x[..., 1:] - x[..., :-1]
        
        a = y > x[..., 1:]
        b = y < x[..., :-1]
        j = np.logical_and(~a, ~b)

        a = a.astype(np.float32)
        b = b.astype(np.float32)
        j = j.astype(np.float32)

        c = (bin*a*np.power(p, 2) + bin*b*np.power(1.0 - p, 2)).sum(axis = -1) + (j*((y - x[..., :-1])*np.power(p, 2) + (x[..., 1:] - y)*np.power(1.0 - p, 2))).sum(axis = -1)
        c = c + (x[..., 0] - y[..., 0])*b[..., 0] + (y[..., 0] - x[..., -1])*a[..., -1]

        return np.reshape(c, shp)

CUDA = torch.cuda.is_available() 

#########################################################

CFG_PATH = "" if len(sys.argv) == 1 else sys.argv[1]

cfg = json.load(open(join(CFG_PATH, "config.json")))

OUTPUT_PATH = cfg["generateMulti"]["outputPath"] 
MODELS_PATH = cfg["generateMulti"]["modelsPath"]
BEST_MODEL_ONLY    = cfg["generateMulti"]["useBestModelOnly"] == "True"
AVERAGE_PARAMETERS = cfg["generateMulti"]["averageParameters"] == "True"
RESIDUALS = cfg["generateMulti"]["residuals"] == "True"

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

X = normalize(X, X_mean, X_std)
Y = normalize(Y, Y_mean, Y_std)
P = normalize(P, P_mean, P_std)

#########################################################

BATCH_SIZE = 512

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

#########################################################

nbins = 51

qstep = 1.0/(nbins + 1)
q = torch.arange(qstep, 1.0, step = qstep, dtype = torch.float32)
q = torch.reshape(q, (1, 1, q.shape[0]))
q = q.cuda() if CUDA else q

print(f"Number of quantiles: {q.shape} |  number of required parameters: 21x{model_distribution.number_of_outputs//21}")

#########################################################

def generate_scores(l, permutation, X, Y, P):

    scores = []

    with torch.no_grad():

        # (229, 730, 21, 51) (229, 730, 21) (229, 4)

        X = np.swapaxes(X, -1, -2)
        shp = X.shape
        X = np.reshape(X, (np.prod(shp[:-1]), shp[-1]))
            
        X[:, l] = X[permutation, l]

        X = np.reshape(X, shp)
        X = np.swapaxes(X, -1, -2)

        bank = Databank(X, Y, P, valid_time, cuda = CUDA)
        dataset = Dataset(bank, bank.index, batch_size = BATCH_SIZE, train = False, cuda = CUDA)

        for i in range(len(dataset)):

            x, p, y, j = dataset[i]
            qtmp = q.expand(y.shape[0], y.shape[1], -1)

            f = []
            for model in models_regression:

                model_distribution.set_parameters(model(x, p))
                f.append(model_distribution.iF(qtmp))

            f = torch.stack(f, dim = 0).mean(dim = 0)

            if RESIDUALS:
                f = (f*Y_std + (x*X_std + X_mean).mean(axis = -1)[..., None]).detach().cpu().numpy()
            else:
                f = f.detach().cpu().numpy()*X_std + X_mean
            y = y.detach().cpu().numpy()

            scores.append(crps(f, y))

            if (i + 1) % 100 == 0:
                print(f"    Computing scores on iteration {i + 1}/{len(dataset)}")

    return np.stack(scores, axis = 0)
            
baseline = generate_scores(0, np.arange(8525670), X, Y, P)

print(baseline.shape)

# np.random.permutation(8525670)
