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

params = {"mathtext.default": "regular"} 
plt.rcParams.update(params)

#########################################################

class EvaluationMetrics(ABC):

    def __init__(self, path, name):

        self.path = path
        self.name = name

    @abstractmethod
    def load(self, raw_forecast):
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

    # Continuous ranked probability score estimation based on Hersbach 2000
    def crps(self, y):

        s0, s1, s2 = y.shape
    
        x = self.get_forecast()
        y = y[..., None]

        # Quantile levels for each input quantiles,
        # treating the quantiles as ensemble members with
        # equal weights, P_2 - P_1 = P_2 - P_1 = ...
        # where P denoted the for this empirical distribution
        step = 1.0/x.shape[-1]
        p = np.arange(step, 1.0, step = step)

        x = np.reshape(x, (s0*s1*s2, x.shape[-1]))
        y = np.reshape(y, (s0*s1*s2, y.shape[-1]))

        # Length of the quantile intervals
        # bin[j] =  [x_2 - x_1, x_3 - x_2, ...]
        bin = x[..., 1:] - x[..., :-1]
        
        # Alpha values to determine bin widths
        # 0 < i < N, where N denotes the number of forecasts
        #
        # Alpha, excluding outliers, is defined as:
        #
        # y > x_(i + 1)       => alpha_i = x_(i + 1) - x_i // y is bigger than the upper bound of this bin
        # x_(i + 1) > y > x_i => alpha_i = y - x_i         // y lies inside the bin
        # y < x_i             => alpha_i = 0               // y is smaller than the lower bound of this bin
        # 
        # Example of a: a[j] = [1,   1,   1,   0,   0,   0,   0]
        #                      [x_2, x_3, x_4, x_5, x_6, x_7, x_8]
        a = y > x[..., 1:]

        # Beta, excluding outliers, is defined as:
        #
        # y > x_(i + 1)       => beta_i = 0               // y is bigger than the upper bound of this bin
        # x_(i + 1) > y > x_i => beta_i = x_(i + 1) - y   // y lies inside the bin
        # y < x_i             => beta_i = x_(i + 1) - x_i // y is smaller than the lower bound of this bin
        # 
        # Example of a: b[j] = [0,   0,   0,   0,   1,   1,   1]
        #                      [x_1, x_2, x_3, x_4, x_5, x_6, x_7]
        b = y < x[..., :-1]

        # Find all the bins that contain y
        # ~a[j] and ~b[j] = [0,   0,   0,   1,   0,   0,   0]
        # In case of an outlier this becomes a zero vector, while either a or b contains all zeros and the remaining all ones
        j = np.logical_and(~a, ~b)

        a = a.astype(np.float32)
        b = b.astype(np.float32)
        j = j.astype(np.float32)

        # Add all the bin contributions where y lies completelly outside the bin
        #
        # c[j] = a_1*(x_2 - x_1)*p_1^2 + a_2*(x_3 - x_2)*p_2^2 + a_3*(x_4 - x_3)*p_3^2 + a_4*(x_5 - x_4)*p_4^2 + ... +
        #        b_5*(x_6 - x_5)*p_5^2 + b_6*(x_7 - x_6)*p_6^2 + b_7*(x_8 - x_7)*p_7^2
        c = (bin*a*np.power(p, 2) + bin*b*np.power(1.0 - p, 2)).sum(axis = -1)

        # We handle the case where y lies inside an interval
        c += (j*((y - x[..., :-1])*np.power(p, 2) + (x[..., 1:] - y)*np.power(1.0 - p, 2))).sum(axis = -1)

        # Finall, add outlier contributions
        c += (x[..., 0] - y[..., 0])*b[..., 0] + (y[..., 0] - x[..., -1])*a[..., -1]

        return np.reshape(c, (s0, s1, s2))

    def quantile_loss(self, y):

        q = self.get_forecast()
        y = y[..., None]

        ql = y - q

        t = np.linspace(0.01, 0.99, q.shape[-1])
        t = np.tile(t, (ql.shape[0], ql.shape[1], ql.shape[2], 1))

        i0 = ql <  0.0
        i1 = ql >= 0.0
        
        ql[i0] *= t[i0] - 1.0
        ql[i1] *= t[i1]

        return ql

    def pit(self, y, y_valid, edges_prob_mass = None):

        r = self.get_forecast()

        c = np.zeros(r.shape[-1] + 1, dtype = np.float32)
        c[0]  = (y < r[..., 0])[y_valid].sum()
        c[-1] = (y > r[..., -1])[y_valid].sum()

        for i in range(1, r.shape[-1]):

            if i == r.shape[-1]:
                c[i] = np.logical_and(y >= r[..., i - 1], y <= r[..., i])[y_valid].sum()
            else:
                c[i] = np.logical_and(y >= r[..., i - 1], y < r[..., i])[y_valid].sum()

        c_tot = y_valid.sum()

        mass = 1/(r.shape[-1] + 1) if not edges_prob_mass else edges_prob_mass
        MRE = ((c[0] + c[-1])/(c_tot*mass*2.0) - 1.0)*100.0

        c = c/y_valid.sum()
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
            Q05, Q95 = np.percentile(d, [ 2, 98])

            IQR = Q75 - Q25

            #WH_TOP = d[d < Q75 + IQR*1.5].max()
            #WH_BOT = d[d > Q25 - IQR*1.5].min()
            WH_TOP = Q95
            WH_BOT = Q05

            res[coverage] = {"med": Q50, "q1": Q25, "q3": Q75, "whislo": WH_BOT, "whishi": WH_TOP}

        return res 

    # Composite intervals sharpness as defined in Bremnes 2019
    def sharpness_composite(self, p = [0.5, 0.88, 0.96], probability_mass = None):
        
        x = self.get_forecast()
        n = x.shape[-1]

        i = 1.0/(1.0 + n) if not probability_mass else probability_mass

        res = {}

        for coverage in p:

            y = int(np.ceil(coverage/i))

            d = np.sort(x[..., 1:] - x[..., :-1], axis = -1)[..., :y].sum(axis = -1).flatten()
            d = d[~np.isnan(d)]

            Q50 = np.median(d)
            Q25, Q75 = np.percentile(d, [25, 75])
            Q05, Q95 = np.percentile(d, [ 2, 98])

            IQR = Q75 - Q25

            #WH_TOP = d[d < Q75 + IQR*1.5].max()
            #WH_BOT = d[d > Q25 - IQR*1.5].min()
            WH_TOP = Q95
            WH_BOT = Q05

            res[coverage] = {"med": Q50, "q1": Q25, "q3": Q75, "whislo": WH_BOT, "whishi": WH_TOP}

        return res

#########################################################

class RawEnsemble(EvaluationMetrics):

    def __init__(self, x, model_altitude, station_altitude, temperature_correction = True):
        super().__init__("", "ECMWF")

        c = (0.65*(model_altitude - station_altitude)/100.0)[:, None, None, None]
        self.x = np.sort(x, axis = -1) if not temperature_correction else np.sort(x + c, axis = -1)
        self.x.astype(np.float32)

    def load(self, raw_forecast):
        pass

    def get_forecast_median(self):
        return np.median(self.x, axis = -1)

class ANET2(EvaluationMetrics):

    def __init__(self, path, name):
        super().__init__(path, name)

    def load(self, raw_forecast):
        self.x = np.load(self.path)

    def get_forecast_median(self):
        return self.x[..., self.x.shape[-1]//2]

class ANET1(EvaluationMetrics):

    def __init__(self, path, name):
        super().__init__(path, name)

    def load(self, raw_forecast):
        self.x = xr.open_dataset(self.path)["t2m"].to_numpy()
        self.sort_members()

    def get_forecast_median(self):
        return self.x[..., self.x.shape[-1]//2] 

class EMOS(EvaluationMetrics):

    def __init__(self, path, name):
        super().__init__(path, name)

    def load(self, raw_forecast):
        self.x = xr.open_dataset(self.path)["t2m"].to_numpy()

        self.x[np.isnan(self.x)] = raw_forecast[np.isnan(self.x)]

        self.sort_members()

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

REFERENCE_MODEL = cfg["evaluation"]["referenceModel"]

#########################################################

data  = load_test_dataset(DATA_PATH)
X, Y, P = data[:3]

alt_m = P[:, 0]
alt, lat, lon = P[:, 1], P[:, 2], P[:, 3]

#########################################################

MODELS = [initialize_model(x, X, alt_m, alt) for x in MODELS]
for m in MODELS:
    m.load(X)

#########################################################

y_valid = np.logical_not(np.isnan(Y))

probability_mass = np.diff(np.linspace(0.01, 0.99, X.shape[-1], dtype = np.float32))[0]

pit       = [m.pit(Y, y_valid, edges_prob_mass = 1/(X.shape[-1] + 1))       for m in MODELS]
shrp_comp = [m.sharpness_composite(probability_mass = probability_mass) for m in MODELS]
shrp      = [m.sharpness()           for m in MODELS]
mae   = [m.median_absolute_error(Y)  for m in MODELS]
crps  = [m.crps(Y)                   for m in MODELS]
bias  = [m.bias(Y)                   for m in MODELS]
qloss = [m.quantile_loss(Y)          for m in MODELS]

import json

stats = {}

for i, model in enumerate(MODELS):

    model_name = model.get_name()

    stats[model_name] = {}
    stats[model_name]["crps_mean"]  = f"{np.nanmean(crps[i]):.3f}"
    stats[model_name]["crps_std"]   = f"{np.nanstd(crps[i]):.3f}"
    stats[model_name]["qloss_mean"] = f"{np.nanmean(qloss[i]):.3f}"
    stats[model_name]["qloss_median_mean"] = f"{np.nanmean(qloss[i][..., qloss[i].shape[-1]//2]):.3f}"
    stats[model_name]["qloss_median_std"]  = f"{np.nanstd(qloss[i][..., qloss[i].shape[-1]//2]):.3f}"
    stats[model_name]["bias_mean"] = f"{np.nanmean(bias[i]):.3f}"
    stats[model_name]["bias_std"]  = f"{np.nanstd(bias[i]):.3f}"
    stats[model_name]["mae"]       = f"{np.nanmean(mae[i]):.3f}"

    for cov in shrp_comp[i]:

        stats[model_name][f"shrp_composite_{cov:.2f}_q50"] = f"{shrp_comp[i][cov]['med']:.3f}"
        stats[model_name][f"shrp_composite_{cov:.2f}_q25"] = f"{shrp_comp[i][cov]['q1']:.3f}"
        stats[model_name][f"shrp_composite_{cov:.2f}_q75"] = f"{shrp_comp[i][cov]['q3']:.3f}"
        stats[model_name][f"shrp_composite_{cov:.2f}_whislo"] = f"{shrp_comp[i][cov]['whislo']:.3f}"
        stats[model_name][f"shrp_composite_{cov:.2f}_whishi"] = f"{shrp_comp[i][cov]['whishi']:.3f}"

    # Compute relative performance improvements

    #stats[model_name]["rel_crps_mean"] = f"{np.nanmean(100*(crps[REFERENCE_MODEL] - crps[i])/crps[REFERENCE_MODEL]):.3f}"
    #stats[model_name]["rel_crps_std"]  = f"{np.nanstd(100*(crps[REFERENCE_MODEL] - crps[i])/crps[REFERENCE_MODEL]):.3f}"

    #stats[model_name]["rel_qloss_median_mean"] = f"{np.nanmean(100*(qloss[REFERENCE_MODEL][..., qloss[i].shape[-1]//2] - qloss[i][..., qloss[i].shape[-1]//2])/qloss[REFERENCE_MODEL][..., qloss[i].shape[-1]//2]):.3f}"
    #stats[model_name]["rel_qloss_median_std"]  = f"{np.nanstd(100*(qloss[REFERENCE_MODEL][..., qloss[i].shape[-1]//2] - qloss[i][..., qloss[i].shape[-1]//2])/qloss[REFERENCE_MODEL][..., qloss[i].shape[-1]//2]):.3f}"
 
    #stats[model_name]["rel_bias_mean"] = f"{np.nanmean(bias[REFERENCE_MODEL] - bias[i]):.3f}"
    #stats[model_name]["rel_bias_std"]  = f"{np.nanstd(bias[REFERENCE_MODEL] - bias[i]):.3f}"

    print(stats[model_name], end = "\n\n")

with open(join(OUTPUT_PATH, "stats.json"), "w") as f:
    json.dump(stats, f)

#########################################################

PLOT_PIT = True 
PLOT_CRPS_BIAS_QSS = not True
PLOT_CRPS_PER_STATION = not True
PLOT_QSS_ALT = not True 
PLOT_CSS_ALT = not True
PLOT_SHARPNESS = not True

#########################################################

colors = [

        (88.0/255.0, 123.0/255.0, 183.0/255.0),
        (169.0/255.0, 214.0/255.0, 124.0/255.0),
        (203.0/255.0, 191.0/255.0, 113.0/255.0),
        (206.0/255.0, 91.0/255.0, 91.0/255.0),
        (102.0/255.0, 153.0/255.0, 102.0/255.0)]

markers = ["o", "v", "^", "h", "X", "D"]

if PLOT_PIT:

    from matplotlib.ticker import FormatStrFormatter

    pit_max = 0.0
    pit_plots = []

    font = {"size"   : 30}    
    matplotlib.rc("font", **font)

    number_of_models = max(int(np.array([k.get_name() != "ECMWF" for k in MODELS]).sum()), 1)

    f, ax = plt.subplots(1, number_of_models, figsize = (10*number_of_models, 10), dpi = 300)
    #prob_mass = np.diff(np.linspace(0.01, 0.99, X.shape[-1], dtype = np.float32))[0]
    prob_mass = 1.0/(X.shape[-1] + 1) 

    offset = 0
    for i, p in enumerate(pit):

        name = MODELS[i].get_name()

        if name == "ECMWF" and len(MODELS) > 1:
            offset += 1
            continue

        a = ax[i - offset] if len(MODELS) > 1 else ax

        nbins = p["pit"].shape[0]

        c = a.hist(np.arange(1, nbins + 1), weights = p["pit"], bins = nbins, label = f"{name}\nMRE: {p['MRE']:.2f}%", color = colors[i], edgecolor = "white")
        a.hlines(prob_mass, 2, nbins - 1, color = "black", linestyle = "dashed")
        #a.hlines(0.01, 1, 2, color = "black")
        #a.hlines(0.01, X.shape[-1], X.shape[-1] + 1, color = "black")
        a.hlines(prob_mass, 1, 2, color = "black")
        a.hlines(prob_mass, X.shape[-1], X.shape[-1] + 1, color = "black")

        pit_max = pit_max if c[0].max() < pit_max else c[0].max()

        a.legend()
        a.set_xlabel("Ranks")

        if i == 0 or name == "ECMWF":
            a.set_ylabel("Density")

        a.set_xticks([5.5, 15.5, 25.5, 35.5, 45.5], labels = [5, 15, 25, 35, 45])

    for a in ax if len(MODELS) > 1 else [ax]:

        #a.set_ylim(0.0, pit_max*1.3)
        a.set_ylim(0.0, 0.1)
        a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))

    f.tight_layout()
    f.savefig(join(OUTPUT_PATH, f"PIT.pdf"), bbox_inches = "tight", format = "pdf")
    plt.close(f)

#########################################################

if PLOT_CRPS_BIAS_QSS:

    font = {"size"   : 30}    
    matplotlib.rc("font", **font)
    from matplotlib.ticker import FormatStrFormatter

    f, ax = plt.subplots(1, 3, figsize = (30, 10), dpi = 300)
  
    a = ax[0]
    for i, c in enumerate(crps):

        c = np.nanmean(c, axis = 1)
        a.plot(c.mean(axis = 0), linewidth = 4, label = MODELS[i].get_name(), color = colors[i], marker = markers[i], markersize = 15)

    a.set_ylim(None, 2.2)
    #a.set_xlabel("Lead time [Hours]")
    a.set_ylabel("CRPS [K]")
    a.set_xticks([0, 4, 8, 12, 16, 20], labels = [0, 24, 48, 72, 96, 120])
    a.grid()
    a.legend()
    a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))

    a = ax[1]
    a.hlines(0, 0, 21, linestyle = "dashed", color = "black")
    a.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    for i, c in enumerate(bias):
        a.plot(c, linewidth = 4, label = MODELS[i].get_name(), color = colors[i], marker = markers[i], markersize = 15)

    #a.set_xlabel("Lead time [Hours]")
    a.set_ylabel("Bias [K]")
    a.set_xticks([0, 4, 8, 12, 16, 20], labels = [0, 24, 48, 72, 96, 120])

    a.grid()
    #a.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2], labels = [-0.2, -0.1, 0.0, 0.1, 0.2])
    a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))

    a = ax[2]
    qloss_reference = np.reshape(qloss[REFERENCE_MODEL], (np.prod(qloss[REFERENCE_MODEL].shape[:-1]), qloss[REFERENCE_MODEL].shape[-1]))
    qloss_reference = np.nanmean(qloss_reference, axis = 0)

    quantile_levels = np.linspace(0.01, 0.99, qloss_reference.shape[0])

    for i, q in enumerate(qloss):
    
        q = np.reshape(q, (np.prod(q.shape[:-1]), q.shape[-1]))
        q = np.nanmean(q, axis = 0)

        v = (1.0 - q/qloss_reference)*100.0

        a.plot(quantile_levels, v, color = colors[i], label = MODELS[i].get_name(), linewidth = 5, linestyle = "dashed" if i == 0 else "solid", marker = markers[i], markersize = 15, markevery = 0.1)

    a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))
    a.grid()
    #a.set_xlabel("Quantile level")
    a.set_ylabel("QSS [Percentage]")
    
    f.tight_layout()
    f.savefig(join(OUTPUT_PATH, "CRPS_BIAS_QSS.pdf"), bbox_inches = "tight", format = "pdf")
    plt.close(f)

#########################################################

if PLOT_CRPS_PER_STATION:

    font = {"size"   : 20}
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
    crps_per_station_best = np.nanmean(crps[0], axis = (1, 2)) 

    crps_per_station_second_best = np.ones(lat.shape[0], dtype = np.float32)*np.inf

    for i, c in enumerate(crps[1:]):

        i = i + 1

        print(f"CRPS {MODELS[i].get_name()}: {np.nanmean(c, axis = (0, 1))}")

        crps_now = np.nanmean(c, axis = (1, 2))

        print(np.any(crps_now == crps_per_station_best))

        idx_now = crps_per_station_best > crps_now

        crps_per_station_second_best[idx_now] = crps_per_station_best[idx_now]
        crps_per_station_idx[idx_now] = i
        crps_per_station_best[idx_now] = crps_now[idx_now]

        idx_now = np.logical_and(crps_per_station_second_best > crps_now, crps_per_station_best < crps_now)
        crps_per_station_second_best[idx_now] = crps_now[idx_now]

    for i in np.unique(crps_per_station_idx):

        idx = crps_per_station_idx == i
        impr_factor = crps_per_station_second_best[idx]/crps_per_station_best[idx]

        print(f"{MODELS[i].get_name()}: mean {impr_factor.mean()} {impr_factor.min()} {impr_factor.max()}")

        a.scatter(x = lon[idx], y = lat[idx], color = colors[i], s = 50, alpha = 0.8, label = f"{MODELS[i].get_name()}: {idx.sum()} cases", marker = markers[i], zorder = 1)

    a.legend(fancybox = True, framealpha = 0.5, markerscale = 2)

    f.savefig(join(OUTPUT_PATH, "crps_per_station.png"), bbox_inches = "tight")
    plt.close(f)

#########################################################

if PLOT_QSS_ALT and len(MODELS) > 1:

    font = {"size": 30}
    matplotlib.rc("font", **font)

    f, ax = plt.subplots(1, 3, figsize = (30, 10), dpi = 300)

    for j, (min_alt, max_alt) in enumerate([(-5, 800), (800, 2000), (2000, 3600)]):

        idx = np.logical_and(alt >= min_alt, alt < max_alt)

        qloss_reference = np.nanmean(qloss[REFERENCE_MODEL][idx], axis = (0, 1, 2))
        quantile_levels = np.linspace(0.01, 0.99, qloss_reference.shape[0])

        a = ax[j]
        a.grid()

        #if max_alt < 3600:
        #    a.set_title(f"{idx.sum()} stations\nAltitude in ({min_alt}, {max_alt}] meters")
        #else:
        #    a.set_title(f"{idx.sum()} stations\nAltitude in ({min_alt}, {max_alt}) meters")

        for i, q in enumerate(qloss):
            
            q = np.nanmean(q[idx], axis = (0, 1, 2))
            v = (1.0 - q/qloss_reference)*100.0

            a.plot(quantile_levels, v, color = colors[i], linewidth = 5, label = MODELS[i].get_name(), linestyle = "dashed" if i == 0 else "solid", marker = markers[i], markersize = 15, markevery = 0.1)
            a.set_xlabel("Quantile level")

        if j == 0:
            a.set_ylabel("QSS [Percentage]")

        a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))
        if (max_alt == 800):
            a.legend(loc = "best")

    f.tight_layout()
    f.savefig(join(OUTPUT_PATH, f"qss_alt.pdf"), bbox_inches = "tight", format = "pdf")

    plt.close(f)

#########################################################

if PLOT_CSS_ALT and len(MODELS) > 1:
    
    font = {"size": 30}
    matplotlib.rc("font", **font)

    f, a = plt.subplots(1, figsize = (20, 10), dpi = 300)
    #a.grid()

    width = 1/(len(MODELS) - 1)
    bins = 7
    dif = np.abs(alt - alt_m)
    idx = np.argsort(dif)
    dif = dif[idx]

    batch_size = int(np.ceil(len(dif)/float(bins)))
    bins = int(np.ceil(len(dif)/batch_size))

    crps_reference = np.nanmean(crps[REFERENCE_MODEL][idx], axis = (1, 2))
    offset = 0

    legend_bplots = []
    legend_labels = []

    for i, c in enumerate(crps):

        if i == REFERENCE_MODEL:
            offset += 1
            continue

        c = np.nanmean(c[idx], axis = (1, 2))
        v = (1.0 - c/crps_reference)*100.0

        legend_labels.append(MODELS[i].get_name())

        for j in range(bins):

            i0 = j*batch_size
            i1 = min((j + 1)*batch_size, len(dif))

            bplt = a.boxplot(v[i0:i1], positions = [2*j + (i - offset)*width + width*0.5], patch_artist = True, showfliers = False, widths = [width], zorder = 3)

            for patch in bplt["boxes"]:
                patch.set_facecolor(colors[i])

            a.set_aspect(0.5*(a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))

            if j == 0:
                legend_bplots.append(bplt["boxes"][0])
   
    a.legend(legend_bplots, legend_labels, loc = "upper left", fontsize = 20)
    a.set_xlabel("Mean and standard deviation of\nthe absolute difference in altitude for each bin [Meters]")
    a.set_ylabel("CRPSS [Percentage]")
    a.grid()

    def f_mean(k):
        return np.mean(dif[k*batch_size:min((k + 1)*batch_size, len(dif))])

    def f_std(k):
        return np.std(dif[k*batch_size:min((k + 1)*batch_size, len(dif))])

    a.set_xticks(list(map(lambda j: 2*j + 0.5, range(bins))), labels = list(map(lambda k: "$\mu$: {:.0f}\n$\sigma$: {:.0f}".format(f_mean(k), f_std(k)), range(bins))), fontsize = 25)
    a.tick_params(axis = "y", labelsize = 25)
    a.hlines(0, 0, 2*bins - 1, linestyle = "dashed", color = colors[REFERENCE_MODEL], linewidth = 4, zorder = 1)

    f.tight_layout()
    f.savefig(join(OUTPUT_PATH, f"css_alt.pdf"), bbox_inches = "tight", format = "pdf")

#########################################################

if PLOT_SHARPNESS:

    from matplotlib.ticker import FormatStrFormatter

    font = {"size": 30}
    matplotlib.rc("font", **font)

    ymin, ymax = None, None

    for coverage in [0.5, 0.88, 0.96]:
        for s in shrp_comp:

            if MODELS[i].get_name() == "ECMWF":
                continue

            ymin = s[coverage]["whislo"] if ymin is None or s[coverage]["whislo"] < ymin else ymin
            ymax = s[coverage]["whishi"] if ymax is None or s[coverage]["whishi"] > ymax else ymax

    f, ax = plt.subplots(1, 3, figsize = (30, 10), dpi = 300)

    for j, coverage in enumerate([0.5, 0.88, 0.96]): 

        a = ax[j] 
        a.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        a.set_title(f"Composite coverage interval {int(coverage*100)}%")
        y_ticks = [ymin]

        for i, s in enumerate(shrp_comp): 

            if MODELS[i].get_name() == "ECMWF":
                continue

            scvr = s[coverage]
            scvr["label"] = MODELS[i].get_name()

            pos = np.arange(1, len(MODELS) + 1) if all([k.get_name() != "ECMWF" for k in MODELS]) else np.arange(1, len(MODELS))
            wdt = 0.25

            a.hlines(scvr["med"], 0.5, pos[i], linestyle = "dashed", linewidth = 1, color = "orange", zorder = 0)
            bplt = a.bxp([scvr], [pos[i]], wdt, showfliers = False, patch_artist = True, zorder = 1)

            for patch in bplt["boxes"]:
                patch.set_facecolor(colors[i])
                patch.set(linewidth = 0)

            a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))

        #a.set_xlim(0.5, None)
        y_ticks += [0.0, 14.0]
        #a.set_yticks(y_ticks, fontsize = 30)
        #a.set_ylim(0.0, 15.0)
        a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))
        a.set_xlabel("Models")

        if j == 0:
            a.set_ylabel("Coverage interval length [K]")

    f.tight_layout()
    f.savefig(join(OUTPUT_PATH, f"SHRP.pdf"), bbox_inches = "tight", format = "pdf")


