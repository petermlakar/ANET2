import torch
import torch.nn as nn
import torch.jit as jit

import numpy as np

class SkipBlock(nn.Module):

    def __init__(self, nfeatures):

        super().__init__()

        self.f = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(nfeatures, nfeatures),
                nn.SiLU())

    def forward(self, x):
        return self.f(x) + x

class ANET2(nn.Module):

    def __init__(self, input_sizes, out_features):

        super().__init__()

        in_features = input_sizes["lead_time"]*2 + input_sizes["number_of_predictors"]

        self.f = nn.Sequential(

                nn.Linear(in_features, 128),
                nn.SiLU(),
               
                SkipBlock(128),

                SkipBlock(128),

                SkipBlock(128),

                SkipBlock(128),

                nn.Linear(128, out_features, dtype = torch.float32))

    def forward(self, x, p):

        m = x.mean(dim = -1, keepdim = True)
        s = x.std(dim = -1, keepdim = True)

        x = torch.flatten(torch.cat([m, s], dim = -1), start_dim = -2, end_dim = -1)

        return self.f(torch.cat([x, p], dim = -1))
 
