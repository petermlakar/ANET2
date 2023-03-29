import torch
import torch.jit as jit
import numpy as np

from os.path import exists, join
from os import mkdir

import sys
from dataset import Databank, Dataset, norm, normalize, denormalize, load_training_dataset, load_validation_dataset
import netCDF4

from time import time as timef

import matplotlib.pyplot as plt

CUDA = torch.cuda.is_available() 

#########################################################

if len(sys.argv) < 3:
    print("Usage: python3 <base path> <model name> <path to data>")
    exit()

PATH = sys.argv[1]
MODEL_PATH = join(PATH, sys.argv[2])
MODEL_NAME = sys.argv[2]
DATA_PATH = sys.argv[3]
DATASET_TYPE = "test"

#### Load training data, remove nan ####

_, _,\
_, _,\
_, _, _, _,\
_, _, _, _,\
X_mean, X_std,\
Y_mean, Y_std,\
P_alt_mean_md, P_alt_std_md,\
P_lat_mean_md, P_lat_std_md,\
P_lon_mean_md, P_lon_std_md,\
P_lnd_mean_md, P_lnd_std_md,\
P_alt_mean_st, P_alt_std_st,\
P_lat_mean_st, P_lat_std_st,\
P_lon_mean_st, P_lon_std_st,\
P_lnd_mean_st, P_lnd_std_st,\
_, _,\
_, _ = load_training_dataset(DATA_PATH)

X, Y,\
P_alt_md, P_lat_md, P_lon_md, P_lnd_md,\
P_alt_st, P_lat_st, P_lon_st, P_lnd_st,\
time, stations,\
valid_time = load_validation_dataset(DATA_PATH)

X = normalize(X, X_mean, X_std)
Y = normalize(Y, Y_mean, Y_std)

P_alt_md = normalize(P_alt_md, P_alt_mean_md, P_alt_std_md)
P_lat_md = normalize(P_lat_md, P_lat_mean_md, P_lat_std_md)
P_lon_md = normalize(P_lon_md, P_lon_mean_md, P_lon_std_md)
P_lnd_md = normalize(P_lnd_md, P_lnd_mean_md, P_lnd_std_md)

P_alt_st = normalize(P_alt_st, P_alt_mean_st, P_alt_std_st)
P_lat_st = normalize(P_lat_st, P_lat_mean_st, P_lat_std_st)
P_lon_st = normalize(P_lon_st, P_lon_mean_st, P_lon_std_st)
P_lnd_st = normalize(P_lnd_st, P_lnd_mean_st, P_lnd_std_st)

#########################################################

BATCH_SIZE = 512
MODEL_PATH = join(MODEL_PATH, f"Model_valid")
EVALUATION_OUTPUT = join(PATH, f"{MODEL_NAME}_generated_output")

if not exists(EVALUATION_OUTPUT):
    mkdir(EVALUATION_OUTPUT)

#########################################################

bank = Databank(X, Y, [P_alt_md, P_alt_st, P_lat_md, P_lat_st, P_lon_md, P_lon_st, P_lnd_md, P_lnd_st], valid_time, cuda = CUDA)

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

print(f"Num quantiles: {q.shape}")

##################################
#x, p, _, _ = D.__getitem__(0)
#torch.onnx.export(M, (x, p), "G.onnx")

##################################

with torch.no_grad():
    
    start_time = timef()

    for i in range(len(D)):

        x, p, y, idx = D.__getitem__(i)
        qtmp = q.expand(y.shape[0], y.shape[1], -1)

        f = M.iF(x, p, qtmp)

        P[idx[0, :], idx[1, :], :, :]  = torch.squeeze(f, dim = -1).detach().cpu().numpy()
        #print(f"{i + 1}/{len(D)}: {y.shape} || {avg_time/(i + 1)}")

    print(f"Execution time time: {timef() - start_time} seconds")


print(P.shape)
print(Y.shape)
print(X.shape)

P = denormalize(P, Y_mean, Y_std)
Y = denormalize(Y, Y_mean, Y_std)
X = denormalize(X, X_mean, X_std)

# Write netCDF4 file #

TIER = 1
INSTITUTION = "ARSO"
EXPERIMENT = "ESSD-benchmark"
MODEL = "ANET"
VERSION = "v2.0"

netcdf = netCDF4.Dataset(join(EVALUATION_OUTPUT, f"{TIER}_{EXPERIMENT}_{INSTITUTION}_{MODEL}_{VERSION}_{DATASET_TYPE}.nc"), mode = "w", format = "NETCDF4_CLASSIC")

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
