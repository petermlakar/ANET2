import torch
import torch.jit as jit
import numpy as np

from os.path import exists, join
from os import mkdir, listdir

import sys
from dataset import Databank, Dataset, norm, standardize, destandardize, load_training_dataset, load_test_dataset 

from time import time as timef

import json

from copy import deepcopy


def crps(x, y):

        s0, s1 = y.shape
   
        y = y[..., None]

        # Quantile levels for each input quantiles,
        # treating the quantiles as ensemble members with
        # equal weights, P_2 - P_1 = P_2 - P_1 = ...
        # where P denoted the for this empirical distribution
        step = 1.0/x.shape[-1]
        p = np.arange(step, 1.0, step = step)

        x = np.reshape(x, (s0*s1, x.shape[-1]))
        y = np.reshape(y, (s0*s1, y.shape[-1]))

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

        return np.reshape(c, (s0, s1))

CUDA = torch.cuda.is_available() 

#########################################################

COST_FUNCTION = "crps"
COMPUTE = False 

if COMPUTE:

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

    if COST_FUNCTION == "ll":

        if RESIDUALS:
            Y = (Y - X.mean(axis = -1))/X_std
        else:
            Y = standardize(Y, X_mean, X_std)

    X = standardize(X, X_mean, X_std)
    P = standardize(P, P_mean, P_std)

    #########################################################

    BATCH_SIZE = 512

    #########################################################

    models_losses = []
    models_regression  = []

    from models.FLOW import Model

    #model_distribution = Model(lead_time = 21, nblocks = 4, nknots = 5)

    for m in listdir(MODELS_PATH):
        
        model_distribution = torch.jit.load(join(MODELS_PATH, m, "model_distribution"), map_location = torch.device("cpu"))
        models_regression.append(torch.jit.load(join(MODELS_PATH, m, "model_regression"), map_location = torch.device("cpu")))
        models_losses.append(np.loadtxt(join(MODELS_PATH, m, "Valid_loss")).min())

        if CUDA:
            models_regression[-1] = models_regression[-1].to("cuda:0")

        models_regression[-1].eval()

    if CUDA:
        model_distribution = model_distribution.to("cuda:0")

    i = np.where(np.array(models_losses) == np.array(models_losses).min())[0][0]

    model_regression = models_regression[i]
    models_losses = models_losses[i]


    print(f"Best only mode: loss -> {models_losses}")

    #########################################################

    nbins = 51

    q = torch.linspace(0.01, 0.99, nbins, dtype = torch.float32)[None, None]
    q = q.cuda() if CUDA else q

    print(f"Number of quantiles: {q.shape} |  number of required parameters: 21x{model_distribution.number_of_outputs//21}")

    #########################################################

    def generate_scores(l, permutation, iX, iY, iP):

        scores = []

        observations = []
        predictions = []

        X = deepcopy(iX)

        with torch.no_grad():

            shp = X.shape

            X = np.reshape(X, (np.prod(shp[:-2]), shp[-2], shp[-1]))
            X[:, l] = X[permutation, l]
            X = np.reshape(X, shp)

            print(f"X is changed: {np.abs(X - iX).sum()/np.prod(X.shape)}")

            bank = Databank(X, iY, iP, valid_time, cuda = CUDA)
            dataset = Dataset(bank, bank.index, batch_size = BATCH_SIZE, train = False, cuda = CUDA)

            for i in range(len(dataset)):

                x, p, y, j = dataset[i]

                if COST_FUNCTION == "crps":

                    model_distribution.set_parameters(model_regression(x, p))
                    f = model_distribution.iF(q.expand(y.shape[0], y.shape[-1], -1)) 

                    if RESIDUALS:
                        f = ((f + x.mean(axis = -1)[..., None])*X_std + X_mean)
                    else:
                        f = f.detach().cpu().numpy()*X_std + X_mean

                    predictions.append(f.detach().cpu().numpy())
                    observations.append(deepcopy(y.detach().cpu().numpy()))

                else:

                    model_distribution.set_parameters(model_regression(x, p))
                    scores.append(model_distribution.nloglikelihood(y).detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    print(f"    Processing {i + 1}/{len(dataset)}")

        c = list(map(lambda k: np.nanmean(crps(k[0], k[1])), zip(predictions, observations)))

        predictions = np.concatenate(predictions, axis = 0)
        observations = np.concatenate(observations, axis = 0)

        c = crps(predictions, observations)

        print(f"Total CRPS: {np.nanmean(c)}")

        return np.nanmean(c, axis = 0)

    #########################################################

    baseline = generate_scores(0, np.arange(167170), X, Y, P)

    print("Baseline: " + "".join(list(map(lambda k: "({:.2f}) ".format(k), baseline))))

    I = np.zeros((22, 21), dtype = np.float32)
    I[0] = baseline

    for l in range(21):

        print(f"\nEstimating relative importance of lead time {l + 1}")

        scores = generate_scores(l, np.random.permutation(167170), X, Y, P)
   
        print("Score: " + " ".join(list(map(lambda l: "{:.2f}".format(l), scores))))

        I[l + 1, :] = scores
        print(f"Importance for lead time {l + 1} = " + "".join(list(map(lambda k: "{:.2f} ".format(k), I[l + 1, :]))))
   
    np.save("importance_crps", I)

else:

    import matplotlib
    import matplotlib.pyplot as plt
    font = {"size"   : 24}    
    matplotlib.rc("font", **font)

    params = {"mathtext.default": "regular"} 
    plt.rcParams.update(params)

    I = np.load("importance_crps.npy")
    I = np.flip((I[1:, :] - I[0, :][None])/I[0, :][None], axis = 0)

    for j in range(21):
        for i in range(21):
            print("{:.2f} ".format(I[j, i]), end = "")
        print()

    f, a = plt.subplots(1, dpi = 300, figsize = (10, 10))

    img = a.imshow(I, cmap = "hot")
    plt.colorbar(img, ax = a, shrink = 0.7125)
    
    a.plot(np.arange(0, I.shape[0]), np.flip(np.arange(0, I.shape[0])), color = "lightskyblue", linestyle = "dashed", alpha = 0.5)

    #a.hlines([20, 16, 12, 8, 4, 0], [0, 0, 0, 0, 0, 0], [0, 4, 8, 12, 16, 20], color = "lightskyblue", linestyle = "dashed", alpha = 0.5)
    a.vlines([0, 4, 8, 12, 16, 20], [20, 20, 20, 20, 20, 20], [20, 16, 12, 8, 4, 0], color = "lightskyblue", linestyle = "dashed", alpha = 0.5)

    a.set_ylabel("Permuted lead time [Hours]")
    a.set_xlabel("Target lead time [Hours]")

    a.set_xticks(np.arange(0, 21, step = 4), labels = np.arange(0, 21, step = 4)*6)
    a.set_yticks(np.arange(0, 21, step = 4), labels = np.flip(np.arange(0, 21, step = 4)*6))

    #a.set_title("Negative log-likelihood relative importance", pad = 30)
    a.set_title("$ANET2_{FLOW}$ lead time importance", pad = 30)

    f.tight_layout()
    f.savefig("importance.pdf", format = "pdf", bbox_inches = "tight")


