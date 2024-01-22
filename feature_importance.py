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

COST_FUNCTION = "ll"

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
        Y = (Y - X.mean(axis = -1))/X_std

    X = standardize(X, X_mean, X_std)
    P = standardize(P, P_mean, P_std)

    #########################################################

    BATCH_SIZE = 512

    #########################################################

    models_losses = []
    models_regression  = []

    from models.FLOW import Model

    model_distribution = Model(lead_time = 21, nblocks = 4, nknots = 5)

    for m in listdir(MODELS_PATH):

        models_regression.append(torch.jit.load(join(MODELS_PATH, m, "model_regression"), map_location = torch.device("cpu")))
        models_losses.append(np.loadtxt(join(MODELS_PATH, m, "Valid_loss")).min())
        #model_distribution = model_distribution if model_distribution is not None else torch.jit.load(join(MODELS_PATH, m, "model_distribution"), map_location = torch.device("cpu"))

        if CUDA:
            models_regression[-1] = models_regression[-1].to("cuda:0")

        models_regression[-1].eval()

    if CUDA:
        model_distribution = model_distribution.to("cuda:0")

    i = np.where(np.array(models_losses) == np.array(models_losses).min())[0][0]

    models_regression = [models_regression[i]]
    models_losses = models_losses[i]

    print(f"Best only mode: loss -> {models_losses}")

    #########################################################

    nbins = 51

    qstep = 1.0/(nbins + 1)
    q = torch.arange(qstep, 1.0, step = qstep, dtype = torch.float32)
    q = torch.reshape(q, (1, 1, q.shape[0]))
    q = q.cuda() if CUDA else q

    print(f"Number of quantiles: {q.shape} |  number of required parameters: 21x{model_distribution.number_of_outputs//21}")

    #########################################################

    def generate_scores(l, permutation, iX, iY, iP):

        scores = []

        X = deepcopy(iX)

        with torch.no_grad():

            shp = X.shape
    
            X = np.reshape(X, (np.prod(shp[:-2]), shp[-2], shp[-1]))
            X[:, l] = X[permutation, l]
            X = np.reshape(X, shp)

            bank = Databank(X, iY, iP, valid_time, cuda = CUDA)
            dataset = Dataset(bank, bank.index, batch_size = BATCH_SIZE, train = False, cuda = CUDA)

            for i in range(len(dataset)):

                x, p, y, j = dataset[i]

                if COST_FUNCTION == "crps":

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

                else:
                    model_distribution.set_parameters(models_regression[-1](x, p))
                    scores.append(model_distribution.nloglikelihood(y).detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    print(f"    Computing scores on iteration {i + 1}/{len(dataset)}")

        if COST_FUNCTION == "crps":
            return np.nanmean(np.concatenate(scores, axis = 0), axis = 0)
        else:
            return np.nanmean(np.concatenate(scores, axis = 0), axis = 0)

    baseline = generate_scores(0, np.arange(167170), X, Y, P)

    print("Baseline: " + "".join(list(map(lambda k: "({:.3f}) ".format(k), baseline))))


    I = np.zeros((21, 21), dtype = np.float32)
    S = []

    for l in range(21):

        print(f"\nEstimating relative importance of lead time {l + 1}")

        scores = generate_scores(l, np.random.permutation(167170), X, Y, P)

        print("Scores: " + "".join(list(map(lambda k: "({:.3f}) ".format(k), scores))))

        if COST_FUNCTION == "crps":
            I[:, l] = np.abs(baseline - scores)/baseline
            print(f"Importance for lead time {l + 1} = " + "".join(list(map(lambda k: "{:.3f} ".format(k), I[:, l]))))
        else:
            S.append(scores)

    if COST_FUNCTION == "ll":
        
        S = np.stack(S, axis = 0)

        a = S.min() - 1.0
        b = np.abs((S[0] - a)/S[0])

        for l in range(21):

            I[:, l] = b*np.abs(S[0] - S[l])/(S[0] - a)
            print(f"Importance for lead time {l + 1} = " + "".join(list(map(lambda k: "{:.3f} ".format(k), I[:, l]))))
    
    np.save("importance", I)

else:
    import matplotlib.pyplot as plt

    I = np.flip(np.load("importance_relative.npy"), axis = 0)
    #I = I/I.sum(axis = 0)[None]

    print(I.mean(axis = 0))

    for j in range(21):
        for i in range(21):
            print("{:.2f} ".format(I[j, i]), end = "")
        print()

    f, a = plt.subplots(1, dpi = 300, figsize = (10, 10))

    a.imshow(I, cmap = "hot")
    
    f.tight_layout()
    f.savefig("importance.pdf", format = "pdf", bbox_inches = "tight")
