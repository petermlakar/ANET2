import torch
import torch.jit as jit
import numpy as np

from os.path import exists, join
from os import mkdir

import sys
from dataset import Databank, Dataset, norm, normalize, denormalize, load_training_dataset, load_test_dataset 
import netCDF4

from time import time as timef

import json

CUDA = torch.cuda.is_available() 

#########################################################

CFG_PATH = "" if len(sys.argv) == 1 else sys.argv[1]

cfg = json.load(open(join(CFG_PATH, "config.json")))

OUTPUT_PATH = cfg["generate"]["outputPath"] 
MODEL_PATH  = cfg["generate"]["modelPath"]
DATA_PATH = cfg["dataPath"]
RESIDUALS = cfg["generate"]["residuals"] == "True"

print(RESIDUALS)

#### Load training data, remove nan ####

_, _,\
_, _,\
_,\
X_mean, X_std,\
Y_mean, Y_std,\
P_mean, P_std,\
_, _,\
_, _ = load_training_dataset(DATA_PATH, MODEL_RESIDUALS = RESIDUALS)

X, Y, P,\
time, stations,\
valid_time = load_test_dataset(DATA_PATH)

X = normalize(X, X_mean, X_std)
Y = normalize(Y, Y_mean, Y_std)
P = normalize(P, P_mean, P_std)

#########################################################

BATCH_SIZE = 512

if not exists(OUTPUT_PATH):
    mkdir(OUTPUT_PATH)

#########################################################

bank = Databank(X, Y, P, valid_time, cuda = CUDA)

D = Dataset(bank, bank.index, batch_size = BATCH_SIZE, train = False, cuda = CUDA)
M = torch.jit.load(MODEL_PATH)
M = M.cuda() if CUDA else M.cpu()
M.eval()

nbins = 51

P = np.zeros(Y.shape + (nbins,), dtype = np.float32)
ENS_C = np.zeros(X.shape, dtype = np.float32)

qstep = 1.0/(nbins + 1)
q = torch.arange(qstep, 1.0, step = qstep, dtype = torch.float32)
q = torch.reshape(q, (1, 1, q.shape[0]))
q = q.cuda() if CUDA else q

print(f"Number of quantiles: {q.shape}")

##################################

with torch.no_grad():
    
    start_time = timef()

    for i in range(len(D)):

        if (i + 1) % 100 == 0:
            print(f"Inference: {i}/{len(D)}")

        x, p, y, idx = D.__getitem__(i)
        qtmp = q.expand(y.shape[0], y.shape[1], -1)

        f = M.iF(x, p, qtmp)
    
        if RESIDUALS:
            P[idx[0, :], idx[1, :], :, :] = (f*Y_std + Y_mean + (x*X_std + X_mean).mean(axis = -1)[..., None]).detach().cpu().numpy()

        else:
            P[idx[0, :], idx[1, :], :, :]  = f.detach().cpu().numpy()*X_std + X_mean

    print(f"Execution time time: {timef() - start_time} seconds")

TIER = 1
INSTITUTION = "ARSO"
EXPERIMENT = "ESSD-benchmark"
MODEL = "ANET"
VERSION = "v2.0"

np.save(join(OUTPUT_PATH, f"{TIER}_{EXPERIMENT}_{INSTITUTION}_{MODEL}_{VERSION}_test"), P)

"""
# Write netCDF4 file #


netcdf = netCDF4.Dataset(join(EVALUATION_OUTPUT, f"{TIER}_{EXPERIMENT}_{INSTITUTION}_{MODEL}_{VERSION}_test.nc"), mode = "w", format = "NETCDF4_CLASSIC")

netcdf.createDimension("station_id", X.shape[0])
netcdf.createDimension("number", q.shape[-1])
netcdf.createDimension("step", X.shape[2])
netcdf.createDimension("time", X.shape[1])

t2m = netcdf.createVariable("t2m", np.float32, ("station_id", "time", "step", "number"))

t2m.institution = INSTITUTION
t2m.tier = TIER
t2m.experiment = EXPERIMENT
t2m.model = MODEL_NAME
t2m.version = VERSION
t2m.output = "quantiles"

t2m[:, :, :, :] = P

netcdf.createVariable("model_altitude", np.float32, ("station_id"), fill_value = np.nan)
netcdf.createVariable("model_latitude", np.float32, ("station_id"), fill_value = np.nan)
netcdf.createVariable("model_longitude", np.float32, ("station_id"), fill_value = np.nan)
netcdf.createVariable("model_land_usage", np.float32, ("station_id"), fill_value = np.nan)

netcdf.createVariable("station_altitude", np.float32, ("station_id"), fill_value = np.nan)
netcdf.createVariable("station_latitude", np.float32, ("station_id"), fill_value = np.nan)
netcdf.createVariable("station_longitude", np.float32, ("station_id"), fill_value = np.nan)
netcdf.createVariable("station_land_usage", np.float32, ("station_id"), fill_value = np.nan)

print(netcdf)

netcdf.close()
"""
