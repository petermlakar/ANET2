import torch
import torch.nn as nn
import torch.jit as jit
import numpy as np

from models.ANET2 import ANET2

class Model(nn.Module):

    def __init__(self, number_of_predictors, lead_time, number_of_stations):

        super().__init__()

        self.lead_time = lead_time

        self.sq   = np.sqrt(2.0*np.pi)
        self.sq_r = np.sqrt(2.0) 
        self.sfp = nn.Softplus()

        # Define parameter regression neural network for mean and variance of normal distribution
        self.parameter_regression = ANET2({"lead_time": lead_time, "number_of_predictors": number_of_predictors}, 2*lead_time, number_of_stations)

    def forward(self, x, p, emb_idx):

        # x: [batch, lead, members]
        # p: [batch, predictors]

        y = self.parameter_regression(x, p, emb_idx)

        p = y.view((y.shape[0], self.lead_time, 2))

        prm_mu = p[:, :, 0]
        prm_sg = p[:, :, 1]

        return prm_mu, self.sfp(prm_sg)

    @jit.ignore
    def loss(self, x, p, f, emb_idx):

        # x: [batch, lead, member]
        # p: [batch, predictors]
        # f: [batch, lead]
        idx = torch.logical_not(torch.isnan(f))

        if idx.sum() == 0:
            return None

        mu, sg = self(x, p, emb_idx)

        mu = mu[idx]
        sg = torch.clamp(sg[idx], min = 1e-6)
        f = f[idx]

        T0 = torch.log(sg)
        T1 = 0.5*torch.pow((mu - f)/sg, 2)

        L = T0 + T1
        return L.mean()

    @jit.export
    def pdf(self, x, p, f, emb_idx):

        # x: [batch, lead, member]
        # p: [batch, predictors]
        # f: [batch, lead]
   
        mu, sg = self(x, p, emb_idx)
        
        return torch.exp(-0.5*torch.pow((f - x)/sg, 2))/(sg*self.sq)

    @jit.export
    def iF(self, x, p, f, emb_idx):

        # x: [batch, lead, members]
        # p: [batch, predictors]
        # f: [batch, lead, quantiles]

        mu, sg = self(x, p, emb_idx)

        mu = mu.view((mu.shape[0], mu.shape[1], 1))
        sg = sg.view((sg.shape[0], sg.shape[1], 1))

        return mu + sg*torch.erfinv(2.0*f - 1.0)*self.sq_r

