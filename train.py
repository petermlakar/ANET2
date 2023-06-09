import numpy as np
import torch
import torch.jit as jit

from os.path import exists, join
from datetime import datetime
from os import mkdir
import sys

from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import Databank, Dataset, norm, normalize, load_training_dataset, load_validation_dataset, denormalize

#########################################################

CUDA = torch.cuda.is_available()

#########################################################

if len(sys.argv) == 1:
    print("Usage: python3 train.py <model type> <path to data> <optional postfix>")
    sys.exit(2)

if sys.argv[1] == "ANET2":
    from models.ANET2 import Model
elif sys.argv[1] == "ANET2_NORM":
    from models.ANET2_NORM import Model
elif sys.argv[1] == "ANET2_BERN":
    from models.ANET2_BERN import Model
else:
    print(f"Invalid model type \"{sys.argv[1]}\" valid types include: (ANET2), (ANET2_NORM), (ANET2_BERN)")

PATH = sys.argv[2]
POSTFIX = "_" + sys.argv[3]

#### Load training data, remove nan ####

X, Y,\
vX, vY,\
P_alt_md, P_lat_md, P_lon_md, P_lnd_md,\
P_alt_st, P_lat_st, P_lon_st, P_lnd_st,\
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
time_train, time_valid = load_training_dataset(PATH)

bank_training   = Databank(X, Y, [P_alt_md, P_alt_st, P_lat_md, P_lat_st, P_lon_md, P_lon_st, P_lnd_md, P_lnd_st], time_train, cuda = CUDA)
bank_validation = Databank(vX, vY, [P_alt_md, P_alt_st, P_lat_md, P_lat_st, P_lon_md, P_lon_st, P_lnd_md, P_lnd_st], time_valid, cuda = CUDA)

#########################################################

N_EPOCHS = 3000
LEARNING_RATE = 1e-3
BATCH_SIZE    = 256
TOLERANCE = 3000

#########################################################

TIMESTAMP = str(datetime.now()).replace(" ", "_").split(":")[0]
OUTPUT_PREFIX = f"{sys.argv[1]}_{LEARNING_RATE}_{BATCH_SIZE}_{TIMESTAMP}{POSTFIX}" # CHANGE THIS APPROPRIATELY BEFORE TRAINING #

BASE_PATH = join(".", OUTPUT_PREFIX)

if not exists(BASE_PATH):
    mkdir(BASE_PATH)

#########################################################

D_train = Dataset(bank_training,   bank_training.index,   batch_size = BATCH_SIZE, cuda = CUDA, train = True)
D_valid = Dataset(bank_validation, bank_validation.index, batch_size = BATCH_SIZE, cuda = CUDA, train = False)

D_train.__on_epoch_end__()
D_valid.__on_epoch_end__()

print(f"Training: {D_train.index.shape}\nValidation: {D_valid.index.shape}")

#########################################################

# 8 covariate + 1 cosine time stamp
M = Model(number_of_predictors = 8 + 1, lead_time = 21)
M = M.cuda() if CUDA else M.cpu()

opt = torch.optim.Adam(M.parameters(), lr = LEARNING_RATE, weight_decay = 1e-6)
sch = ReduceLROnPlateau(opt, factor = 0.9, patience = 10)

best_train_loss = None
best_valid_loss = None

train_losses = np.zeros(N_EPOCHS, dtype = np.float32)
valid_losses = np.zeros(N_EPOCHS, dtype = np.float32)

for e in range(N_EPOCHS):

    train_loss = 0.0
    valid_loss = 0.0

    c_train = 0
    c_valid = 0

    #### Train an epoch ####
    M.train()
    for i in range(len(D_train)):

        x, p, y, _ = D_train.__getitem__(i)

        loss = M.loss(x, p, y)

        if loss is None:
            continue

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item()
        c_train += 1

    D_train.__on_epoch_end__()

    #### Test an epoch ####
    
    M.eval()
    with torch.no_grad():

        valid_loss = 0.0
        for i in range(len(D_valid)):

            x, p, y, _ = D_valid.__getitem__(i)
            loss = M.loss(x, p, y).item()

            if loss is None:
                continue

            valid_loss += loss
            c_valid += 1

    D_valid.__on_epoch_end__()

    sch.step(valid_loss)

    #### Record best losses and save model

    train_loss /= c_train
    valid_loss /= c_valid

    print(f"{e + 1}/{N_EPOCHS}: TLoss {train_loss} VLoss {valid_loss}\n")
    
    train_losses[e] = train_loss
    valid_losses[e] = valid_loss

    if best_train_loss is None or train_loss < best_train_loss:
        
        best_train_loss = train_loss

        jit.save(jit.script(M.cpu()), join(BASE_PATH, f"Model_train"))
        if CUDA:
            M.cuda()
    
    if best_valid_loss is None or valid_loss < best_valid_loss:
        
        best_valid_loss = valid_loss

        jit.save(jit.script(M.cpu()), join(BASE_PATH, f"Model_valid"))
        if CUDA:
            M.cuda()

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


