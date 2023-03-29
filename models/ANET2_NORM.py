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

class Model(nn.Module):

    def __init__(self, number_of_predictors, lead_time):

        super().__init__()

        self.lead_time = lead_time

        self.sq   = np.sqrt(2.0*np.pi)
        self.sq_r = np.sqrt(2.0) 
        self.sfp = nn.Softplus()

        # Define regression block

        self.R = nn.Sequential(

                nn.Linear(number_of_predictors + lead_time*2, 128),
                nn.SiLU(),
               
                SkipBlock(128),

                SkipBlock(128),

                SkipBlock(128),

                SkipBlock(128),

                nn.Linear(128, 2*lead_time, dtype = torch.float32))

    def forward(self, x, p):

        # x: [batch, lead, members]
        # p: [batch, predictors]

        m = x.mean(dim = -1, keepdim = True)
        s = x.std(dim = -1, keepdim = True)

        x = torch.flatten(torch.cat([m, s], dim = -1), start_dim = -2, end_dim = -1)
        x = torch.cat([x, p], dim = -1)

        Y = self.R(x)

        p = Y.view((Y.shape[0], self.lead_time, 2))

        prm_mu = p[:, :, 0]
        prm_sg = p[:, :, 1]

        return prm_mu + torch.squeeze(m, dim = -1), self.sfp(prm_sg + torch.squeeze(s, dim = -1))

    @jit.ignore
    def loss(self, x, p, f):

        # x: [batch, lead, member]
        # p: [batch, predictors]
        # f: [batch, lead]
        idx = torch.logical_not(torch.isnan(f))

        if idx.sum() == 0:
            return None

        mu, sg = self(x, p)

        mu = mu[idx]
        sg = torch.clamp(sg[idx], min = 1e-6)
        f = f[idx]

        T0 = torch.log(sg)
        T1 = 0.5*torch.pow((mu - f)/sg, 2)

        L = T0 + T1
        return L.mean()

    @jit.export
    def pdf(self, x, p, f):

        # x: [batch, lead, member]
        # p: [batch, predictors]
        # f: [batch, lead]
   
        mu, sg = self(x, p)
        
        return torch.exp(-0.5*torch.pow((f - x)/sg, 2))/(sg*self.sq)

    @jit.export
    def iF(self, x, p, f):

        # x: [batch, lead, members]
        # p: [batch, predictors]
        # f: [batch, lead, quantiles]

        mu, sg = self(x, p)

        mu = mu.view((mu.shape[0], mu.shape[1], 1))
        sg = sg.view((sg.shape[0], sg.shape[1], 1))

        return mu + sg*torch.erfinv(2.0*f - 1.0)*self.sq_r

