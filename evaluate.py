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

        """
        import properscoring as ps
        s0, s1, s2 = y.shape

        x = self.get_forecast().reshape(s0*s1*s2, 51)
        y = y.reshape(s0*s1*s2)

        
        G = x.shape[0]//100
        print(x.shape)

        c = []
        for i in range(G):
            c.append(ps.crps_ensemble(y[i*100:(i + 1)*100], x[i*100:(i + 1)*100]))

            if (i + 1) % 1000 == 0:
                print(f"{(i + 1)/G}")
        
        c.append(ps.crps_ensemble(y[G*100:], x[G*100:]))
        c = np.nanmean(np.concatenate(c, axis = 0).reshape(s0, s1, s2), axis = (0, 1)) 

        print("CRPS according to properscoring: ", " ".join(list(map(lambda l: "{:.2f}".format(l), c))))
        """

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

        c = np.reshape(c, (s0, s1, s2))

        return c

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
        c[0]  = (y < r[..., 0])[y_valid].sum()
        c[-1] = (y > r[..., -1])[y_valid].sum()

        for i in range(1, r.shape[-1]):

            if i == r.shape[-1]:
                c[i] = np.logical_and(y >= r[..., i - 1], y <= r[..., i])[y_valid].sum()
            else:
                c[i] = np.logical_and(y >= r[..., i - 1], y < r[..., i])[y_valid].sum()

        c = c/y_valid.sum()

        MRE = ((c[0] + c[-1])*(r.shape[-1] + 1)/2.0 - 1.0)*100.0

        return {"MRE": MRE, "pit": c}

    def bias(self, y):

        x = self.get_forecast_median()
        
        return np.nanmean((x - y), axis = (0, 1))

    # Central intervals sharpness
    def sharpness(self, p = [0.5, 0.88, 0.96]):
        
        x = self.get_forecast()
        n = x.shape[-1]

        i = 1.0/(1 + n)

        res = {}

        for coverage in p:

            y = int(np.ceil(coverage/i)) 
            d = (x[..., n//2 + y//2] - x[..., n//2 - y//2]).flatten()
            d = d[~np.isnan(d)]

            Q50 = np.median(d)
            Q25, Q75 = np.percentile(d, [25, 75])

            IQR = Q75 - Q25

            WH_TOP = d[d < Q75 + IQR*1.5].max()
            WH_BOT = d[d > Q25 - IQR*1.5].min()

            res[coverage] = {"med": Q50, "q1": Q25, "q3": Q75, "whislo": WH_BOT, "whishi": WH_TOP}

        return res 

    # Composite intervals sharpness as defined in Bremnes 2019
    def sharpness_composite(self, p = [0.5, 0.88, 0.96]):
        
        x = self.get_forecast()
        n = x.shape[-1]

        i = 1.0/(1.0 + n)

        res = {}

        for coverage in p:

            y = int(np.ceil(coverage/i))
            d = np.sort(x[..., 1:] - x[..., :-1], axis = -1)[..., :y].sum(axis = -1).flatten()
            d = d[~np.isnan(d)]

            Q50 = np.median(d)
            Q25, Q75 = np.percentile(d, [25, 75])

            IQR = Q75 - Q25

            WH_TOP = d[d < Q75 + IQR*1.5].max()
            WH_BOT = d[d > Q25 - IQR*1.5].min()

            res[coverage] = {"med": Q50, "q1": Q25, "q3": Q75, "whislo": WH_BOT, "whishi": WH_TOP}

        return res

#########################################################

class RawEnsemble(EvaluationMetrics):

    def __init__(self, x, model_altitude, station_altitude, temperature_correction = True):
        super().__init__("", "ECMWF")

        c = (0.6*(model_altitude - station_altitude)/100.0)[:, None, None, None]
        self.x = np.sort(x, axis = -1) if not temperature_correction else np.sort(x + c, axis = -1)

    def load(self):
        pass

    def get_forecast_median(self):
        return np.median(self.x, axis = -1)

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

    def get_forecast_median(self):
        return self.x[..., self.x.shape[-1]//2] 

class EMOS(EvaluationMetrics):

    def __init__(self, path, name):
        super().__init__(path, name)

    def load(self):
        self.x = xr.open_dataset(self.path)["t2m"]

        print(self.x.coords)

        self.x = self.x.to_numpy()

        exit()

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
        c = c/valid.sum()

        MRE = ((c[0] + c[-1])*(r.shape[-1] + 1)/2.0 - 1.0)*100.0

        return {"MRE": MRE, "pit": c}

class DVINE(EvaluationMetrics):
    
    def __init__(self, path, name):
        super().__init__(path, name)

    def load(self):
        self.x = xr.open_dataset(self.path)["t2m"].to_numpy()

    def get_forecast_median(self):
        return self.x[:, :, :, self.x.shape[-1]//2]

def initialize_model(model, x = None, model_altitude = None, station_altitude = None):

    path = model["path"]
    name = model["name"]

    match model["type"]:

        case "ECMWF":
            return RawEnsemble(x, model_altitude, station_altitude)

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

MODELS = [initialize_model(x, X, alt_m, alt) for x in MODELS]
for m in MODELS:
    m.load()

#########################################################

y_valid = np.logical_not(np.isnan(Y))

pit       = [m.pit(Y, y_valid)       for m in MODELS]
shrp_comp = [m.sharpness_composite() for m in MODELS]
shrp      = [m.sharpness()           for m in MODELS]
mae   = [m.median_absolute_error(Y)  for m in MODELS]
crps  = [m.crps(Y)                   for m in MODELS]
bias  = [m.bias(Y)                   for m in MODELS]
qloss = [m.quantile_loss(Y)          for m in MODELS]

stats = []

for i, model in enumerate(MODELS):

    stats.append("Model {} scores:\n     Continuous ranked probability score: {:.3f}\n     Quantile loss: {:.3f}\n     Median absolute error: {:.3f}\n     Bias: {:.3f}\n     Central interval sharpness:\n".format(
                  model.get_name(), np.nanmean(crps[i]), 
                  np.nanmean(qloss[i]), np.nanmean(mae[i]), np.nanmean(bias[i])))

    for cov in shrp[i]:
        stats[-1] += "         Coverage {:.2f}: q50 {:.3f} | q25 {:.3f} q75 {:.3f} | whilo {:.3f} whihi {:.3f}\n".format(
                      cov, 
                      shrp[i][cov]["med"], shrp[i][cov]["q1"], shrp[i][cov]["q3"], shrp[i][cov]["whislo"], shrp[i][cov]["whishi"])

    stats[-1] += "     Composite interval sharpness:\n"
    for cov in shrp_comp[i]:
        stats[-1] += "         Coverage {:.2f}: q50 {:.3f} | q25 {:.3f} q75 {:.3f} | whilo {:.3f} whihi {:.3f}\n".format(
                      cov, 
                      shrp_comp[i][cov]["med"], shrp_comp[i][cov]["q1"], shrp_comp[i][cov]["q3"], shrp_comp[i][cov]["whislo"], shrp_comp[i][cov]["whishi"])

    print(stats[-1])

with open(join(OUTPUT_PATH, "stats.txt"), "w") as f:
    for s in stats:
        f.write(s)

#########################################################

PLOT_PIT  = True 
PLOT_CRPS = False
PLOT_BIAS = False
PLOT_CRPS_PER_STATION = False
PLOT_QSS = False
PLOT_QSS_ALT = False
PLOT_CSS_ALT = False 
PLOT_SHARPNESS = True

#########################################################

colors = [
        (169.0/255.0, 214.0/255.0, 124.0/255.0),
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

        nbins = p["pit"].shape[0]

        c = pit_plots[i][1].hist(np.arange(1, nbins + 1), weights = p["pit"], bins = np.arange(1, 54), label = "{}\nMRE: {:.2f}%".format(MODELS[i].get_name(), p["MRE"]), color = colors[i], edgecolor = "black")
        c = c[0]

        pit_plots[i][1].hlines(1.0/nbins, 1, nbins, color = "black", linestyle = "dashed")
        pit_max = pit_max if c.max() < pit_max else c.max()

        #pit_plots[i][1].grid()
        pit_plots[i][1].legend()

        pit_plots[i][1].set_xlabel("Bins")
        pit_plots[i][1].set_ylabel("Density")

        pit_plots[i][1].set_xticks([5.5, 15.5, 25.5, 35.5, 45.5], labels = [5, 15, 25, 35, 45])

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

        a.plot(np.nanmean(c, axis = (0, 1)), linewidth = 4, label = MODELS[i].get_name(), color = colors[i], marker = markers[i], markersize = 15)

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

        a.plot(c, linewidth = 4, label = MODELS[i].get_name(), color = colors[i], marker = markers[i], markersize = 15)

        a.set_xlabel("Lead time [6 hours]")
        a.set_ylabel("Bias [Temperature in K]")

    a.set_ylim(-0.25, 0.25)
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

        a.scatter(x = lon[idx], y = lat[idx], color = colors[i], s = 40, alpha = 0.8, transform = ccrs.PlateCarree(), label = f"{MODELS[i].get_name()}: {idx.sum()} cases", marker = markers[i], zorder = 1)

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

        a.plot(quantile_levels, v, color = colors[i], label = MODELS[i].get_name(), linewidth = 5, linestyle = "dashed" if i == 0 else "solid", marker = markers[i], markersize = 15, markevery = 0.1)

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

            a.plot(quantile_levels, v, color = colors[i], linewidth = 5, label = MODELS[i].get_name(), linestyle = "dashed" if i == 0 else "solid", marker = markers[i], markersize = 15, markevery = 0.1)

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

    crps_reference = np.nanmean(crps[0][idx], axis = (1, 2))

    f, a = plt.subplots(1, figsize = (10, 10), dpi = 300)
    a.grid()

    for i, c in enumerate(crps):

        c = np.nanmean(c[idx], axis = (1, 2))
        v = (1.0 - c/crps_reference)*100.0

        v = medfilt(v, kernel_size = 15)

        a.plot(alt, v, color = colors[i], linewidth = 5, label = MODELS[i].get_name(), linestyle = "dashed" if i == 0 else "solid", marker = markers[i], markersize = 15, markevery = 0.1)

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

    for i, s in enumerate(shrp): 

        f, a = plt.subplots(1, dpi = 300, figsize = (10, 10))

        a.set_title(MODELS[i].get_name()) 

        y_ticks = [0]

        for j, coverage in enumerate([0.5, 0.88, 0.96]): 

            scvr = s[coverage]
            scvr["label"] = " {}%".format(int(coverage*100))

            pos = np.arange(1, 4)
            wdt = 0.25
            y_ticks.append(scvr["med"])

            a.hlines(scvr["med"], 0.5, pos[j], linestyle = "dashed", linewidth = 1, label = f"{int(coverage*100)}% median: " + "{:.2f}".format(scvr["med"]), color = "orange")
            bplt = a.bxp([scvr], [pos[j]], wdt, showfliers = False, patch_artist = True)

            for patch in bplt["boxes"]:
                patch.set_facecolor(colors[i])

            a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))

        a.set_xlim(0.5, None)
        y_ticks.append(np.ceil(ymax))
        a.set_yticks(y_ticks)

        if i == 1:
            a.set_xlabel("Central coverage interval [percentage]")
        if i == 0:
            a.set_ylabel("Coverage interval length [K]")
        #a.legend(prop={"size": 20})

        f.tight_layout()
        f.savefig(join(OUTPUT_PATH, f"cnt_shrp_{MODELS[i].get_name()}.pdf"), bbox_inches = "tight", format = "pdf")

    for i, s in enumerate(shrp_comp): 

        f, a = plt.subplots(1, dpi = 300, figsize = (10, 10))

        a.set_title(MODELS[i].get_name()) 

        y_ticks = [0]

        for j, coverage in enumerate([0.5, 0.88, 0.96]): 

            scvr = s[coverage]
            scvr["label"] = " {}%".format(int(coverage*100))

            pos = np.arange(1, 4)
            wdt = 0.25
            y_ticks.append(scvr["med"])

            a.hlines(scvr["med"], 0.5, pos[j], linestyle = "dashed", linewidth = 1, label = f"{int(coverage*100)}% median: " + "{:.2f}".format(scvr["med"]), color = "orange")
            bplt = a.bxp([scvr], [pos[j]], wdt, showfliers = False, patch_artist = True)

            for patch in bplt["boxes"]:
                patch.set_facecolor(colors[i])

            a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))

        a.set_xlim(0.5, None)
        y_ticks.append(np.ceil(ymax))
        a.set_yticks(y_ticks)
        a.set_xlabel("Composite coverage interval [percentage]")

        if i == 0:
            a.set_ylabel("Coverage interval length [K]")
        #a.legend(prop={"size": 20})

        f.tight_layout()
        f.savefig(join(OUTPUT_PATH, f"cmp_shrp_{MODELS[i].get_name()}.pdf"), bbox_inches = "tight", format = "pdf")


