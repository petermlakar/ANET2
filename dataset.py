import xarray as xr
import numpy as np
import torch

from datetime import datetime

from os.path import join

#### Dataset definition ####

class Databank():

    def __init__(self, X, Y, P, T, cuda = True):

        self.X = torch.tensor(X, dtype = torch.float32)
        self.Y = torch.tensor(Y, dtype = torch.float32)
        self.P = torch.tensor(P)
        self.T = torch.tensor(T, dtype = torch.float32)
        self.n_samples = X.shape[0]*X.shape[1]
        self.n_members = X.shape[-1]

        print(f"P shape: {self.P.shape}")

        self.index = np.stack(np.meshgrid(np.arange(0, X.shape[0]), np.arange(0, X.shape[1])), axis = 0)
        self.index = np.reshape(self.index, (self.index.shape[0], np.prod(self.index.shape[1:])))

        if cuda:

            self.X = self.X.cuda()
            self.Y = self.Y.cuda()
            self.P = self.P.cuda()
            self.T = self.T.cuda()

class Dataset(torch.utils.data.Dataset):

    def __init__(self, databank, index, n_predictors = 8, batch_size = 32, cuda = True, train = True):
        
        super().__init__()

        self.databank = databank
        self.index = index

        self.batch_size = batch_size
        self.n_samples = self.index.shape[1]
        self.n_members = self.databank.n_members
        self.train = train

        self.size = int(np.ceil(self.n_samples/batch_size))

        lead_time = self.databank.X.shape[2]

        # X_batch: [batch_size, lead_time, n_members]
        # P_batch: [batch_size, n_predictors]
        # Y_batch: [batch_size, lead_time]

        self.X_batch = torch.zeros((batch_size, lead_time, self.n_members), dtype = torch.float32)
        self.P_batch = torch.zeros((batch_size, self.databank.P.shape[1] + 1))
        self.Y_batch = torch.zeros((batch_size, lead_time), dtype = torch.float32)

        if cuda:

            self.X_batch = self.X_batch.cuda()
            self.Y_batch = self.Y_batch.cuda()
            self.P_batch = self.P_batch.cuda()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
       
        i0 = idx*self.batch_size
        i1 = min((idx + 1)*self.batch_size, self.n_samples)

        i = self.index[:, i0:i1]
        t = i1 - i0
        
        self.X_batch[:t, :, :] = self.databank.X[i[0], i[1], :, :]
        self.Y_batch[:t, :]    = self.databank.Y[i[0], i[1], :]
        self.P_batch[:t, :-1]  = self.databank.P[i[0], :]
        self.P_batch[:t, -1]   = self.databank.T[i[0], i[1]]

        if self.train:
            r_m = np.random.permutation(self.n_members)
            X = self.X_batch[:, :, r_m]

        else:
            X = self.X_batch

        return X[:t], self.P_batch[:t], self.Y_batch[:t], i
    
    def shuffle(self):
        
        idx = np.random.permutation(self.n_samples)

        self.index[0, :] = self.index[0, idx]
        self.index[1, :] = self.index[1, idx]

def norm(T, remove_nan = False):

    if remove_nan:
        m = T[~np.isnan(T)].mean()
        s = T[~np.isnan(T)].std()
    else:
        m = T.mean()
        s = T.std()

    return (T - m)/s, m, s

def normalize(T, m, s):
    return (T - m)/s

def denormalize(T, m, s):
    return T*s + m

def load_training_dataset(path, residuals = False):

    X = xr.open_dataarray(join(path, "ESSD_benchmark_training_data_forecasts.nc"))
    Y = xr.open_dataarray(join(path, "ESSD_benchmark_training_data_observations.nc")).to_numpy()

    time = np.array([str(x).split("T")[0] for x in X.coords["time"].to_numpy()])
    stations = [str(x) for x in X.coords["station_name"].to_numpy()]

    P_alt_md = X.coords["model_altitude"].to_numpy().astype(np.float32)
    P_alt_st = X.coords["station_altitude"].to_numpy().astype(np.float32)
    P_lat_st = X.coords["station_latitude"].to_numpy().astype(np.float32)
    P_lon_st = X.coords["station_longitude"].to_numpy().astype(np.float32)

    P = np.stack([P_alt_md, P_alt_st, P_lat_st, P_lon_st], axis = 1)

    X = np.squeeze(X.to_numpy())

    assert X.shape[-1] == 11

    def get_idx(V):
        return np.logical_not(np.all(np.all(np.isnan(np.reshape(V, (V.shape[0], V.shape[1], np.prod(V.shape[2:])))), axis = -1), axis = 0))

    def collapse(V):
        return np.reshape(V, (V.shape[0], V.shape[1]*V.shape[2]) + V.shape[3:])

    X = np.pad(X, [(0, 0), (0, 0), (0, 0), (0, 1), (0, 0)])

    for i, t in enumerate(time):
        datetime_object = datetime.strptime(t, "%Y-%m-%d")
        X[:, i, :, -1, :] = np.cos(np.pi*2.0*datetime_object.timetuple().tm_yday/365.0)

    vX = X[:, :, -2:]
    vY = Y[:, :, -2:]

    tX = X[:, :, 0:-2]
    tY = Y[:, :, 0:-2]

    x_s2015 = vX[:, :104, 0]
    y_s2015 = vY[:, :104, 0]

    x_s2016_0 = vX[:, 104:, 0]
    y_s2016_0 = vY[:, 104:, 0]

    x_s2016_1 = vX[:, :104, 1]
    y_s2016_1 = vY[:, :104, 1]

    tX = collapse(tX)
    tY = collapse(tY)

    tX = np.concatenate([tX, x_s2015], axis = 1)
    tY = np.concatenate([tY, y_s2015], axis = 1)

    vX = np.concatenate([x_s2016_0, x_s2016_1], axis = 1)
    vY = np.concatenate([y_s2016_0, y_s2016_1], axis = 1)

    tidx = get_idx(tX)
    vidx = get_idx(vX)

    tX = tX[:, tidx]
    tY = tY[:, tidx]

    vX = vX[:, vidx]
    vY = vY[:, vidx]

    time_train = tX[:, :, -1, 0]
    time_valid = vX[:, :, -1, 0]
    tX = tX[:, :, :-1]
    vX = vX[:, :, :-1]

    P_mean = P.mean(axis = 0)
    P_std  = P.std(axis = 0)

    P = (P - P_mean[None])/P_std[None]

    if residuals:

       tY = tY - tX.mean(axis = -1)
       vY = vY - vX.mean(axis = -1)

       tX, tX_mean, tX_std = norm(tX)
       tY, tY_mean, tY_std = norm(tY, remove_nan = True)

       vX = (vX - tX_mean)/tX_std
       vY = (vY - tY_mean)/tY_std

    else:

        tX, tX_mean, tX_std = norm(tX)
        tY = (tY - tX_mean)/tX_std

        vX = (vX - tX_mean)/tX_std
        vY = (vY - tX_mean)/tX_std

        tY_mean, tY_std = (tX_mean, tX_std)

    return tX, tY,\
    vX, vY,\
    P, \
    tX_mean, tX_std,\
    tY_mean, tY_std,\
    P_mean[None], P_std[None],\
    time, np.array(stations),\
    time_train, time_valid

def load_test_dataset(path):
    
    X = xr.open_dataarray(join(path, "ESSD_benchmark_test_data_forecasts.nc"))
    Y = xr.open_dataarray(join(path, "ESSD_benchmark_test_data_observations.nc")).to_numpy()

    time = [str(x).split("T")[0] for x in X.coords["time"].to_numpy()]
    stations = [str(x) for x in X.coords["station_name"].to_numpy()]

    P_alt_md = X.coords["model_altitude"].to_numpy().astype(np.float32)
    P_alt_st = X.coords["station_altitude"].to_numpy().astype(np.float32)
    P_lat_st = X.coords["station_latitude"].to_numpy().astype(np.float32)
    P_lon_st = X.coords["station_longitude"].to_numpy().astype(np.float32)

    P = np.stack([P_alt_md, P_alt_st, P_lat_st, P_lon_st], axis = 1)

    X = np.squeeze(X.to_numpy())

    T = np.zeros((1, len(time)), dtype = np.float32)

    for i, t in enumerate(time):
        datetime_object = datetime.strptime(t, "%Y-%m-%d")
        T[0, i] = np.cos(np.pi*2.0*datetime_object.timetuple().tm_yday/365.0)

    T = np.repeat(T, X.shape[0], axis = 0)

    return X, Y, P,\
    np.array(time), np.array(stations), T


