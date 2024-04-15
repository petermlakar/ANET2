import numpy as np
import torch
import torch.jit as jit

from os.path import exists, join
from datetime import datetime
from os import mkdir
import sys

from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import Databank, Dataset, load_training_dataset

import json

from time import time

#########################################################

CUDA = torch.cuda.is_available()

CFG_PATH = "" if len(sys.argv) == 1 else sys.argv[1]

cfg = json.load(open(join(CFG_PATH, "config.json")))

DATA_PATH = cfg["dataPath"]
MODEL_TYPE = cfg["training"]["modelType"]

RESIDUALS = cfg["training"]["residuals"]

#########################################################

LEAD_TIME = 21
NUMBER_OF_STATIONS = 229

from models.ANET2 import ANET2

match MODEL_TYPE:
    
    case "FLOW":
        from models.FLOW import Model
        model_distribution = Model(LEAD_TIME)
    case "NORM":
        from models.NORM import Model
        model_distribution = Model(LEAD_TIME)
    case "BERN":
        from models.BERN import Model
        model_distribution = Model(LEAD_TIME, degree = 8, monotone = False)
    case _:

        print(f"Invalid model type in config {MODEL_TYPE}...\nSupported types are: FLOW, NORM, BERN")
        exit()

model_regression = ANET2({"lead_time": LEAD_TIME, "number_of_predictors": 4 + 1}, model_distribution.number_of_outputs, NUMBER_OF_STATIONS)

#########################################################
# Load training data, remove nan

POSTFIX = ("_" + cfg["training"]["postfix"] if cfg["training"]["postfix"] != "" else "")

X, Y,\
vX, vY,\
P,\
X_mean, X_std,\
Y_mean, Y_std,\
P_mean, P_std,\
_, _,\
time_train, time_valid = load_training_dataset(DATA_PATH, RESIDUALS)

bank_training   = Databank(X, Y, P, time_train, cuda = CUDA)
bank_validation = Databank(vX, vY, P, time_valid, cuda = CUDA)

#########################################################

N_EPOCHS = 600
LEARNING_RATE = 1e-3
BATCH_SIZE    = 256
TOLERANCE = 30

#########################################################

TIMESTAMP = str(datetime.now()).replace(" ", "_").split(":")[0]
OUTPUT_PREFIX = f"{MODEL_TYPE}_{LEARNING_RATE}_{BATCH_SIZE}_{TIMESTAMP}{POSTFIX}" # CHANGE THIS APPROPRIATELY BEFORE TRAINING #

BASE_PATH = join(".", OUTPUT_PREFIX)

if not exists(BASE_PATH):
    mkdir(BASE_PATH)

#########################################################

dataset_train = Dataset(bank_training,   bank_training.index,   batch_size = BATCH_SIZE, cuda = CUDA, train = True)
dataset_valid = Dataset(bank_validation, bank_validation.index, batch_size = BATCH_SIZE, cuda = CUDA, train = False)

dataset_train.shuffle()
dataset_valid.shuffle()

print(f"Training: {dataset_train.index.shape}\nValidation: {dataset_valid.index.shape}")

#########################################################

if CUDA:
    model_distribution = model_distribution.to("cuda:0")
    model_regression   = model_regression.to("cuda:0")

#########################################################

opt = torch.optim.AdamW(model_regression.parameters(), lr = LEARNING_RATE, weight_decay = 1e-6)
sch = ReduceLROnPlateau(opt, factor = 0.9, patience = 3)

#########################################################

best_valid_loss = None

train_losses = np.zeros(N_EPOCHS, dtype = np.float32)
valid_losses = np.zeros(N_EPOCHS, dtype = np.float32)

for e in range(N_EPOCHS):

    train_loss = 0.0
    valid_loss = 0.0

    c_train = 0
    c_valid = 0

    #### Train an epoch ####
    model_regression.train()
    for i in range(len(dataset_train)):

        x, p, y, j = dataset_train[i]

        parameters = model_regression(x, p, j[0])
        model_distribution.set_parameters(parameters)
        loss = model_distribution.loss(y)

        if loss is None:
            continue

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item()
        c_train += 1

        if (i + 1) % 500 == 0:
            print(f"    Training loss {i + 1}/{len(dataset_train)}: {train_loss/c_train}")

    #### Validate an epoch ####
    
    model_regression.eval()
    with torch.no_grad():

        valid_loss = 0.0
        for i in range(len(dataset_valid)):

            x, p, y, j = dataset_valid[i]

            parameters = model_regression(x, p, j[0])
            model_distribution.set_parameters(parameters)
            loss = model_distribution.loss(y).item()

            if loss is None:
                continue

            valid_loss += loss
            c_valid += 1

            if (i + 1) % 50 == 0:
                print(f"    Validation loss {i + 1}/{len(dataset_valid)}: {valid_loss/c_valid}")

    dataset_train.shuffle()
    dataset_valid.shuffle()

    #### Record best losses and save model

    train_loss /= c_train
    valid_loss /= c_valid

    sch.step(valid_loss)

    print(f"{e + 1}/{N_EPOCHS}: TLoss {train_loss} VLoss {valid_loss}")

    train_losses[e] = train_loss
    valid_losses[e] = valid_loss
   
    if best_valid_loss is None or valid_loss < best_valid_loss:
        
        best_valid_loss = valid_loss

        jit.save(jit.script(model_regression.cpu()),   join(BASE_PATH, f"model_regression"))
        jit.save(jit.script(model_distribution.cpu()), join(BASE_PATH, f"model_distribution"))

        if CUDA:
            model_regression.to("cuda:0")
            model_distribution.to("cuda:0")

    np.savetxt(join(BASE_PATH, f"Train_loss"), train_losses[0:e + 1])
    np.savetxt(join(BASE_PATH, f"Valid_loss"), valid_losses[0:e + 1])

    #### Early stopping ####

    e_cntr = e + 1
    if e_cntr > TOLERANCE:
        
        min_val_loss = valid_losses[0:e_cntr].min()
        print("Early stopping check: ({:.5f})".format(min_val_loss))

        if not (min_val_loss in valid_losses[e_cntr - TOLERANCE:e_cntr]):

            print(f"Early stopping at epoch {e_cntr}")
            break

    print(f"Best validation loss: {best_valid_loss}\n")

