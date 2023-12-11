import xarray as xr
import numpy as np
from scipy.special import erf

from functools import reduce

from os.path import exists, join
from os import mkdir
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as mticker

font = {"size"   : 24}    
matplotlib.rc("font", **font)

from dataset import load_test_dataset, load_training_dataset
import json

from abc import ABC, abstractmethod

#########################################################

class EvaluationMetrics(ABC):

    def __init__(self, path, name):

        self.path = path
        self.name = name

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_forecast_median(self):
        pass

    #########################################################

    def get_forecast(self):
        return self.x

    def get_name(self):
        return self.name

    def sort_members(self):
        self.x.sort(axis = -1) 

    #########################################################

    def median_absolute_error(self, y):
        return np.abs(self.get_forecast_median() - y)


    # Continuous ranked probability score estimation
    # based on Hersbach 2000, where the model forecast,
    # be it quantiles or samples form the predictive
    # distribution, is treated as an ensemble forecast,
    # each member having an equal weight.
    def crps(self, y):

        s0, s1, s2 = y.shape
    
        x = self.get_forecast()
        y = y[..., None]

        p = np.arange(1.0/x.shape[-1], 1.0, step = 1.0/x.shape[-1])[None]

        x = np.reshape(x, (s0*s1*s2, x.shape[-1]))
        y = np.reshape(y, (s0*s1*s2, y.shape[-1]))

        bin = x[..., 1:] - x[..., :-1]
        
        a = y > x[..., 1:]
        b = y < x[..., :-1]
        j = np.logical_and(~a, ~b)

        a = a.astype(np.float32)
        b = b.astype(np.float32)
        j = j.astype(np.float32)

        c = (bin*a*np.power(p, 2) + bin*b*np.power(1.0 - p, 2)).sum(axis = -1) + (j*((y - x[..., :-1])*np.power(p, 2) + (x[..., 1:] - y)*np.power(1.0 - p, 2))).sum(axis = -1)
        c = c + (x[..., 0] - y[..., 0])*b[..., 0] + (y[..., 0] - x[..., -1])*a[..., -1]

        return np.reshape(c, (s0, s1, s2))

        """
        I = 1000

        print(c[I])
        print(tmp[I])
        print(d0[I])
        print(d1[I])

        for i in range(1, x.shape[-1]):
            print("[{:.2f}, {:.2f}] ".format(x[I, i - 1], x[I, i]) + f"alpha: {a[I, i - 1]} beta: {b[I, i - 1]} j: {j[I, i - 1]}" + " -> observation: {:.2f} | p: {:.3f}".format(y[I, 0], p[0, i - 1]))
        """

    def quantile_loss(self, y):

        q = self.get_forecast()
        y = y[..., None]

        ql = y - q

        t = np.arange(1.0/(q.shape[-1] + 1), 1.0, step = 1.0/(q.shape[-1] + 1)).reshape((1, 1, 1, ql.shape[-1]))

        t = np.repeat(t, ql.shape[0], axis = 0)
        t = np.repeat(t, ql.shape[1], axis = 1)
        t = np.repeat(t, ql.shape[2], axis = 2)

        ql[ql <  0] *= (t[ql <  0] - 1.0)
        ql[ql >= 0] *= t[ql >= 0]

        return ql

    def pit(self, y, y_valid):

        r = self.get_forecast()

        c = np.zeros(r.shape[-1] + 1, dtype = np.float32)
        c[0]  = (y <= r[..., 0])[y_valid].sum()
        c[-1] = (y >  r[..., -1])[y_valid].sum()

        for i in range(1, r.shape[-1]):
            c[i] = np.logical_and(y >= r[..., i - 1], y < r[..., i])[y_valid].sum()
        c = c/y_valid.sum()

        return c

    def bias(self, y):

        x = self.get_forecast_median()
        
        return np.nanmean((x - y), axis = (0, 1))



#########################################################

class ANET2(EvaluationMetrics):

    def __init__(self, path, name):
        super().__init__(path, name)

    def load(self):
        self.x = np.load(self.path)

    def get_forecast_median(self):
        return self.x[..., self.x.shape[-1]//2]

class ANET1(EvaluationMetrics):

    def __init__(self, path, name):
        super().__init__(path, name)

    def load(self):
        self.x = xr.open_dataset(self.path)["t2m"].to_numpy()
        self.sort_members()

class EMOS(EvaluationMetrics):

    def __init__(self, path, name):
        super().__init__(path, name)

    def load(self):
        self.x = xr.open_dataset(self.path)["t2m"].to_numpy()
        self.sort_members()

    def get_forecast_median(self):
        return self.x[:, :, :, self.x.shape[-1]//2]

    def pit(self, y, y_valid):
        
        r = self.get_forecast()

        valid = np.logical_and(np.logical_not(np.all(np.isnan(r), axis = -1)), y_valid)

        c = np.zeros(r.shape[-1] + 1, dtype = np.float32)
        c[0]  = (y <= r[..., 0])[valid].sum()
        c[-1] = (y >  r[..., -1])[valid].sum()

        for i in range(1, r.shape[-1]):
            c[i] = np.logical_and(y >= r[..., i - 1], y < r[..., i])[valid].sum()

        return c/valid.sum()

class DVINE(EvaluationMetrics):
    
    def __init__(self, path, name):
        super().__init__(path, name)

    def load(self):
        self.x = xr.open_dataset(self.path)["t2m"].to_numpy()

    def get_forecast_median(self):
        return self.x[:, :, :, self.x.shape[-1]//2]

def initialize_model(model):

    path = model["path"]
    name = model["name"]

    match model["type"]:
        
        case "ANET2":
            return ANET2(path, name)

        case "ANET1":
            return ANET1(path, name)

        case "EMOS":
            return EMOS(path, name)

        case "DVINE":
            return DVINE(path, name)

#########################################################

CFG_PATH = "" if len(sys.argv) == 1 else sys.argv[1]

cfg = json.load(open(join(CFG_PATH, "config.json")))

DATA_PATH = cfg["dataPath"]
MODELS = cfg["evaluation"]

#########################################################

data  = load_test_dataset(DATA_PATH)
X, Y = data[0], data[1]

alt_m = data[2]
alt, lat, lon = data[6], data[7], data[8]

#########################################################

MODELS = [initialize_model(x) for x in MODELS]
for m in MODELS:
    m.load()

#########################################################

y_valid = np.logical_not(np.isnan(Y))

mae   = [m.median_absolute_error(Y) for m in MODELS]
pit   = [m.pit(Y, y_valid)          for m in MODELS]
crps  = [m.crps(Y)                  for m in MODELS]
bias  = [m.bias(Y)                  for m in MODELS]
qloss = [m.quantile_loss(Y)         for m in MODELS]

for i, model in enumerate(MODELS):
    print("Model {} scores:\n     Continuous ranked probability score: {:.3f}\n     Quantile loss: {:.3f}\n     Median absolute error: {:.3f}\n     Bias: {:.3f}".format(model.get_name(), np.nanmean(crps[i]), np.nanmean(qloss[i]), np.nanmean(mae[i]), np.nanmean(bias[i])))

#########################################################

PLOT_PIT  = True
PLOT_CRPS = True

#########################################################

colors = [
        (203.0/255.0, 191.0/255.0, 113.0/255.0),
        (206.0/255.0, 91.0/255.0, 91.0/255.0),
        (88.0/255.0, 123.0/255.0, 183.0/255.0),
        (102.0/255.0, 153.0/255.0, 102.0/255.0)]

if PLOT_PIT:

    pit_max = 0.0
    pit_plots = []

    font = {"size"   : 30}    
    matplotlib.rc("font", **font)

    for i, p in enumerate(pit):

        pit_plots.append((plt.subplots(1, figsize = (10, 10), dpi = 300)))

        nbins = MODELS[i].get_forecast().shape[-1] + 1

        c = pit_plots[i][1].hist(np.arange(nbins), weights = p, bins = nbins, label = MODELS[i].get_name(), color = colors[i])
        c = c[0]

        pit_plots[i][1].hlines(1.0/nbins, 0, nbins - 1, color = "black", linestyle = "dashed")
        pit_max = pit_max if c.max() < pit_max else c.max()

        pit_plots[i][1].grid()
        pit_plots[i][1].legend()

        pit_plots[i][1].set_xlabel("Bins")
        pit_plots[i][1].set_ylabel("Density")

    for i in range(len(pit)):

        pit_plots[i][1].set_ylim(0.0, pit_max*1.3)
        pit_plots[i][1].set_aspect((pit_plots[i][1].get_xlim()[1] - pit_plots[i][1].get_xlim()[0])/(pit_plots[i][1].get_ylim()[1] - pit_plots[i][1].get_ylim()[0]))

        pit_plots[i][0].tight_layout()
        pit_plots[i][0].savefig(f"PIT_{MODELS[i].get_name()}.pdf", bbox_inches = "tight", format = "pdf")
        plt.close(pit_plots[i][0])

#########################################################

if PLOT_CRPS:

    font = {"size"   : 30}    
    matplotlib.rc("font", **font)

    f, a = plt.subplots(1, figsize = (10, 10), dpi = 300)
    for i, c in enumerate(crps):

        a.plot(np.nanmean(c, axis = (0, 1)), linewidth = 4, label = MODELS[i].get_name(), color = colors[i])

        a.set_xlabel("Lead time [6 hours]")
        a.set_ylabel("CRPS")

    a.grid()
    a.legend()
    a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))

    f.tight_layout()
    f.savefig("CRPS.pdf", bbox_inches = "tight", format = "pdf")
    plt.close(f)

exit()

#########################################################

f, a = plt.subplots(1, figsize = (10, 10), dpi = 300)
a.hlines(0, 0, 21, linestyle = "dashed", color = "black")

for i, c in enumerate(bias):

    a.plot(c, linewidth = 4, label = models[i].get_name(), color = colors[i])

    a.set_xlabel("Lead time [6 hours]")
    a.set_ylabel("Bias [Temperature in K]")

a.set_ylim(-0.5, 0.5)
a.grid()
#a.legend()
a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))

f.tight_layout()
f.savefig("rbias.pdf", bbox_inches = "tight", format = "pdf")
plt.close(f)

##############################

font = {"size"   : 24}    
matplotlib.rc("font", **font)

import cartopy.crs as ccrs
import cartopy

f, a = plt.subplots(1, figsize = (10, 10), dpi = 300, subplot_kw = {"projection": ccrs.PlateCarree()})

a.add_feature(cartopy.feature.LAND,    edgecolor = "black", facecolor = (245.0/255.0, 249.0/255.0, 245.0/255.0))
a.add_feature(cartopy.feature.OCEAN,   edgecolor = "black", facecolor = (181.0/255.0, 201.0/255.0, 225.0/255.0))
a.add_feature(cartopy.feature.LAKES,   edgecolor = "black")
a.add_feature(cartopy.feature.BORDERS, edgecolor = "black")

a.set_extent([2.5, 10.5, 45.75, 53.5])
a.coastlines()

crps_per_station_idx  = np.zeros(lat.shape[1], dtype = np.int32)
crps_per_station_best = np.zeros(lat.shape[1], dtype = np.float32) 

for i, c in enumerate(crps):

    if i == 0:
        crps_per_station_idx[:] = i
        crps_per_station_best[:] = np.nanmean(c, axis = (1, 2))
    else:
        crps_now = np.nanmean(c, axis = (1, 2))

        idx_now = crps_per_station_best > crps_now

        crps_per_station_idx[idx_now] = i
        crps_per_station_best[idx_now] = crps_now[idx_now]

for i in np.unique(crps_per_station_idx):

    idx = crps_per_station_idx == i

    a.scatter(x = lon[0, idx], y = lat[0, idx], color = colors[i], s = 40, alpha = 0.8, transform = ccrs.PlateCarree(), label = f"{models[i].get_name()}: {idx.sum()} cases")

a.legend(fancybox = True, framealpha = 0.5, markerscale = 2)

f.savefig("crps_per_station.png", bbox_inches = "tight")

plt.close(f)

##############################

font = {"size"   : 30}
matplotlib.rc("font", **font)

f, a = plt.subplots(1, figsize = (10, 10), dpi = 300)

q_base = np.nanmean(qloss[-1], axis = (0, 1, 2))
quantile_levels = np.arange(1.0/(q_base.size + 1), 1.0, step = 1.0/(q_base.size + 1))

for i, q in enumerate(qloss):
    if i < len(qloss) - 1:
        a.plot(quantile_levels, (1.0 - np.nanmean(q, axis = (0, 1, 2))/q_base)*100.0, color = colors[i], label = models[i].get_name(), linewidth = 5)
    else:
        a.plot(quantile_levels, (1.0 - np.nanmean(q, axis = (0, 1, 2))/q_base)*100.0, color = colors[i], label = models[i].get_name(), linewidth = 5, linestyle = "dashed")

a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))
a.grid()
#a.legend()

a.set_xlabel("Quantile level")
a.set_ylabel("QSS [percentage]")

f.tight_layout()
f.savefig("qss.pdf", bbox_inches = "tight", format = "pdf")
plt.close(f)

##############################

font = {"size": 30}
matplotlib.rc("font", **font)

print(f"{alt.min()} {alt.max()}")

q_base = np.nanmean(qloss[-1], axis = (1, 2))

for (min_alt, max_alt) in [(-5, 800), (800, 2000), (2000, 3600)]:

    idx = np.logical_and(alt >= min_alt, alt < max_alt)[0, :]

    q_base_alt = q_base[idx, :].mean(axis = 0)

    f, a = plt.subplots(1, figsize = (10, 10), dpi = 300)
    a.set_ylim(-1.0, 21.0)
    a.grid()

    if max_alt < 3600:
        a.set_title(f"{idx.sum()} stations\nAltitude in ({min_alt}, {max_alt}] meters")
    else:
        a.set_title(f"{idx.sum()} stations\nAltitude in ({min_alt}, {max_alt}) meters")

    for i, q in enumerate(qloss):
        q_local = np.nanmean(q, axis = (1, 2))[idx, :].mean(axis = 0)

        if i < len(qloss) - 1:
            a.plot(quantile_levels, 100.0*(1.0 - q_local/q_base_alt), color = colors[i], linewidth = 5, label = models[i].get_name())
        else:
            a.plot(quantile_levels, 100.0*(1.0 - q_local/q_base_alt), color = colors[i], linewidth = 5, linestyle = "dashed", label = models[i].get_name())

    a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))
    if (max_alt == 800):
        a.legend(loc = "best")#, bbox_to_anchor = (-0.1, 1.0))
    a.set_xlabel("Quantile level")
    a.set_ylabel("QSS [percentage]")

    f.savefig(f"qss_alt_{max_alt}.pdf", bbox_inches = "tight", format = "pdf")

    plt.close(f)

##############################

from scipy.signal import medfilt

dif = np.abs(alt - alt_m)[0]
alt_idx = np.argsort(dif)

f, a = plt.subplots(1, figsize = (10, 10), dpi = 300)

a.grid()

for i, c in enumerate(crps):

    y_raw = np.nanmean(c[alt_idx], axis = (1, 2))

    y = medfilt(y_raw, kernel_size = 15)
    ry = np.polyfit(dif[alt_idx], y, deg = 2) 

    a.plot(dif[alt_idx], y,  color = colors[i], linewidth = 5, label = models[i].get_name())
    #a.plot(dif[alt_idx], (dif[alt_idx]**2)*ry[0] + dif[alt_idx]*ry[1] + ry[2], color = colors[i], linewidth = 5, label = models[i].get_name())

a.legend(fancybox = True, framealpha = 0.5, markerscale = 40)
a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))

a.set_xlabel("Absolute difference in altitude [meters]")
a.set_ylabel("CRPS")

f.savefig(f"crps_alt.pdf", bbox_inches = "tight", format = "pdf")
plt.close(f)



