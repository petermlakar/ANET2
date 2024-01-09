import xarray as xr
import numpy as np
from scipy.special import erf
from scipy.signal import medfilt

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

    # Continuous ranked probability score estimation based on Hersbach 2000, where the model forecast,
    # be it quantiles or samples form the predictive distribution, is treated as an ensemble forecast,
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

    def quantile_loss(self, y):

        q = self.get_forecast()
        y = y[..., None]

        ql = y - q

        t = np.arange(1.0/(q.shape[-1] + 1), 1.0, step = 1.0/(q.shape[-1] + 1)).reshape((1, 1, 1, ql.shape[-1]))
        t = np.tile(t, (ql.shape[0], ql.shape[1], ql.shape[2], 1))

        i0 = ql <  0.0
        i1 = ql >= 0.0
        
        ql[i0] *= t[i0] - 1.0
        ql[i1] *= t[i1]

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

    def sharpness(self, p = [0.5, 0.88, 0.96]):
        
        x = self.get_forecast()
        n = x.shape[-1]

        i = 1.0/(1 + n)
        q = np.arange(i, 1.0, step = i, dtype = np.float32)

        res = {}

        for coverage in p:

            y = int(np.ceil(coverage/i)) 
            d = (x[..., n//2 + y//2] - x[..., n//2 - y//2]).flatten()
    
            Q50 = np.median(d)
            Q25, Q75 = np.percentile(d, [25, 75])

            print(f"{coverage} -> {Q50} / {Q25} {Q75}")

            IQR = Q75 - Q25

            WH_TOP = d[d < Q75 + IQR*1.5].max()
            WH_BOT = d[d > Q25 - IQR*1.5].min()

            res[coverage] = {"med": Q50, "q1": Q25, "q3": Q75, "whislo": WH_BOT, "whishi": WH_TOP}
        
        print()

        return res 

    # Sharpness as defined in Bremnes 2019
    def sharpness_composite(self):
        pass

#########################################################

class RawEnsemble(EvaluationMetrics):

    def __init__(self, x):
        super().__init__("", "Raw ensemble")

        self.x = x

    def load(self):
        pass

    def get_forecast_median(self):
        return np.median(x, axis = -1)

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
MODELS      = cfg["evaluation"]["models"]
OUTPUT_PATH = cfg["evaluation"]["outputPath"] 

#########################################################

data  = load_test_dataset(DATA_PATH)
X, Y, P = data[:3]

alt_m = P[:, 0]
alt, lat, lon = P[:, 1], P[:, 2], P[:, 3]

#########################################################

MODELS = [initialize_model(x) for x in MODELS]
for m in MODELS:
    m.load()

#########################################################

y_valid = np.logical_not(np.isnan(Y))

shrp  = [m.sharpness() for m in MODELS]

mae   = [m.median_absolute_error(Y) for m in MODELS]
pit   = [m.pit(Y, y_valid)          for m in MODELS]
crps  = [m.crps(Y)                  for m in MODELS]
bias  = [m.bias(Y)                  for m in MODELS]
qloss = [m.quantile_loss(Y)         for m in MODELS]

stats = []

for i, model in enumerate(MODELS):

    stats.append("Model {} scores:\n     Continuous ranked probability score: {:.3f}\n     Quantile loss: {:.3f}\n     Median absolute error: {:.3f}\n     Bias: {:.3f}\n".format(model.get_name(), np.nanmean(crps[i]), np.nanmean(qloss[i]), np.nanmean(mae[i]), np.nanmean(bias[i])))
    print(stats[-1])

with open(join(OUTPUT_PATH, "stats.txt"), "w") as f:
    for s in stats:
        f.write(s)

#########################################################

PLOT_PIT  = True
PLOT_CRPS = True
PLOT_BIAS = True
PLOT_CRPS_PER_STATION = True
PLOT_QSS = True
PLOT_QSS_ALT = True
PLOT_CSS_ALT = True
PLOT_SHARPNESS = True

#########################################################

colors = [
        (203.0/255.0, 191.0/255.0, 113.0/255.0),
        (206.0/255.0, 91.0/255.0, 91.0/255.0),
        (88.0/255.0, 123.0/255.0, 183.0/255.0),
        (102.0/255.0, 153.0/255.0, 102.0/255.0)]

markers = ["o", "v", "^", "h", "X", "D"]

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
        pit_plots[i][0].savefig(join(OUTPUT_PATH, f"PIT_{MODELS[i].get_name()}.pdf"), bbox_inches = "tight", format = "pdf")
        plt.close(pit_plots[i][0])

#########################################################

if PLOT_CRPS:

    font = {"size"   : 30}    
    matplotlib.rc("font", **font)

    f, a = plt.subplots(1, figsize = (10, 10), dpi = 300)
    for i, c in enumerate(crps):

        a.plot(np.nanmean(c, axis = (0, 1)), linewidth = 4, label = MODELS[i].get_name(), color = colors[i])

        a.set_xlabel("Lead time [6 hours]")
        a.set_ylabel("CRPS [K]")

    a.grid()
    a.legend()
    a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))

    f.tight_layout()
    f.savefig(join(OUTPUT_PATH, "CRPS.pdf"), bbox_inches = "tight", format = "pdf")
    plt.close(f)

#########################################################

if PLOT_BIAS:

    f, a = plt.subplots(1, figsize = (10, 10), dpi = 300)
    a.hlines(0, 0, 21, linestyle = "dashed", color = "black")

    for i, c in enumerate(bias):

        a.plot(c, linewidth = 4, label = MODELS[i].get_name(), color = colors[i])

        a.set_xlabel("Lead time [6 hours]")
        a.set_ylabel("Bias [Temperature in K]")

    a.set_ylim(-0.5, 0.5)
    a.grid()
    a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))

    f.tight_layout()
    f.savefig(join(OUTPUT_PATH, "BIAS.pdf"), bbox_inches = "tight", format = "pdf")
    plt.close(f)

#########################################################

if PLOT_CRPS_PER_STATION:

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

    crps_per_station_idx  = np.zeros(lat.shape[0], dtype = np.int32)
    crps_per_station_best = np.zeros(lat.shape[0], dtype = np.float32) 

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

        a.scatter(x = lon[idx], y = lat[idx], color = colors[i], s = 40, alpha = 0.8, transform = ccrs.PlateCarree(), label = f"{MODELS[i].get_name()}: {idx.sum()} cases", marker = markers[i])

    a.legend(fancybox = True, framealpha = 0.5, markerscale = 2)

    f.savefig(join(OUTPUT_PATH, "crps_per_station.png"), bbox_inches = "tight")

    plt.close(f)

#########################################################

if PLOT_QSS:

    font = {"size": 30} 
    matplotlib.rc("font", **font)

    f, a = plt.subplots(1, figsize = (10, 10), dpi = 300)

    qloss_reference = np.reshape(qloss[0], (np.prod(qloss[0].shape[:-1]), qloss[0].shape[-1]))
    qloss_reference = np.nanmean(qloss_reference, axis = 0)

    quantile_levels = np.arange(1.0/(qloss_reference.shape[-1] + 1), 1.0, step = 1.0/(qloss_reference.shape[-1] + 1))

    for i, q in enumerate(qloss):
    
        q = np.reshape(q, (np.prod(q.shape[:-1]), q.shape[-1]))
        q = np.nanmean(q, axis = 0)

        v = (1.0 - q/qloss_reference)*100.0

        a.plot(quantile_levels, v, color = colors[i], label = MODELS[i].get_name(), linewidth = 5, linestyle = "dashed" if i == 0 else "solid")

    a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))
    a.grid()
    a.legend()

    a.set_xlabel("Quantile level")
    a.set_ylabel("QSS [percentage]")

    f.tight_layout()
    f.savefig(join(OUTPUT_PATH, "qss.pdf"), bbox_inches = "tight", format = "pdf")
    plt.close(f)

#########################################################

if PLOT_QSS_ALT:

    font = {"size": 30}
    matplotlib.rc("font", **font)


    for (min_alt, max_alt) in [(-5, 800), (800, 2000), (2000, 3600)]:

        idx = np.logical_and(alt >= min_alt, alt < max_alt)

        qloss_reference = np.nanmean(qloss[0][idx], axis = (0, 1, 2))
        quantile_levels = np.arange(1.0/(qloss_reference.shape[-1] + 1), 1.0, step = 1.0/(qloss_reference.shape[-1] + 1))

        f, a = plt.subplots(1, figsize = (10, 10), dpi = 300)
        a.grid()

        if max_alt < 3600:
            a.set_title(f"{idx.sum()} stations\nAltitude in ({min_alt}, {max_alt}] meters")
        else:
            a.set_title(f"{idx.sum()} stations\nAltitude in ({min_alt}, {max_alt}) meters")

        for i, q in enumerate(qloss):
            
            q = np.nanmean(q[idx], axis = (0, 1, 2))
            v = (1.0 - q/qloss_reference)*100.0

            a.plot(quantile_levels, v, color = colors[i], linewidth = 5, label = MODELS[i].get_name(), linestyle = "dashed" if i == 0 else "solid")

        a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))
        if (max_alt == 800):
            a.legend(loc = "best")
        a.set_xlabel("Quantile level")
        a.set_ylabel("QSS [percentage]")

        f.tight_layout()
        f.savefig(join(OUTPUT_PATH, f"qss_alt_{max_alt}.pdf"), bbox_inches = "tight", format = "pdf")

        plt.close(f)

#########################################################

if PLOT_CSS_ALT:
    
    font = {"size": 30}
    matplotlib.rc("font", **font)

    idx = np.argsort(alt)
    alt = alt[idx]

    #ens = RawEnsemble(np.sort(X[idx], axis = -1))
    #ens_crps = ens.crps(Y[idx])

    crps_reference = np.nanmean(crps[0][idx], axis = (1, 2))

    f, a = plt.subplots(1, figsize = (10, 10), dpi = 300)
    a.grid()

    for i, c in enumerate(crps):

        c = np.nanmean(c[idx], axis = (1, 2))
        v = (1.0 - c/crps_reference)*100.0

        v = medfilt(v, kernel_size = 15)

        a.plot(alt, v, color = colors[i], linewidth = 5, label = MODELS[i].get_name(), linestyle = "dashed" if i == 0 else "solid")

    a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))
    a.set_xlabel("Station altitude\n[meters above sea level]")
    a.set_ylabel("CRPS Skill Score [Percentage]")

    f.tight_layout()
    f.savefig(join(OUTPUT_PATH, f"css_alt.pdf"), bbox_inches = "tight", format = "pdf")

#########################################################

if PLOT_SHARPNESS:

    font = {"size": 30}
    matplotlib.rc("font", **font)

    ymin, ymax = None, None

    for coverage in [0.5, 0.88, 0.96]:
        for s in shrp:

            ymin = s[coverage]["whislo"] if ymin is None or s[coverage]["whislo"] < ymin else ymin
            ymax = s[coverage]["whishi"] if ymax is None or s[coverage]["whishi"] > ymax else ymax


    f, ax = plt.subplots(1, 3, dpi = 300, figsize = (30, 10))

    for j, coverage in enumerate([0.5, 0.88, 0.96]): 

        a = ax[j]

        pos = np.arange(1, len(MODELS) + 1, step = 1)
        wdt = 0.25

        for i, s in enumerate(shrp): 

            s = s[coverage]
            s["label"] = MODELS[i].get_name()

            bplt = a.bxp([s], [pos[i]], wdt, showfliers = False, patch_artist = True)

            for patch in bplt["boxes"]:
                patch.set_facecolor(colors[i])

        #a.set_ylim(ymin - 0.1, ymax + 0.1)
        a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))
        a.set_title(f"Centered {int(coverage*100)}% interval coverage")

        if j == 1:
            a.set_xlabel("Models")

        if j == 0:
            a.set_ylabel("Coverage interval length [K]")

    f.tight_layout()
    f.savefig(join(OUTPUT_PATH, f"sharpness.pdf"), bbox_inches = "tight", format = "pdf")

