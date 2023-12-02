import torch
import torch.nn as nn
import torch.jit as jit

import numpy as np

import time

import math

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

    def __init__(self, number_of_predictors, lead_time, positive_support = False, monotone = False):

        super().__init__()

        self.lead_time = lead_time
        self.sfp = nn.Softplus()

        self.degree = 12
        self.nquantiles = 100

        # Enforce a positive support for the Bernstein quanilte function
        # and enforce the monotonicity of the coefficients which in turn
        # results in the monotonicity of the quantile function.
        self.positive_support = positive_support
        self.monotone = monotone

        # Bernstein quantile function coefficient regression neural network
        self.R = nn.Sequential(

                nn.Linear(number_of_predictors + lead_time*2, 128),
                nn.SiLU(),
               
                SkipBlock(128),

                SkipBlock(128),

                SkipBlock(128),

                SkipBlock(128),

                nn.Linear(128, (self.degree + 1)*lead_time, dtype = torch.float32))

        ########################################################################
        # Compute binomial coefficients and the remaining required parameters
        
        step = 1.0/(self.nquantiles + 1)
        self.q = torch.arange(step, 1.0, step = step, requires_grad = False)

        self.w = torch.ones(self.degree + 1, dtype = torch.float32, requires_grad = False)
        for i in torch.arange(0, self.degree + 1, dtype = torch.int32):
            self.w[i] = math.comb(self.degree, i)
        self.j = torch.arange(0, self.degree + 1, dtype = torch.float32, requires_grad = False)

        self.q = self.q.view((1, self.nquantiles, 1))

        self.j = nn.Parameter(torch.unsqueeze(torch.unsqueeze(self.j, dim = 0), dim = 0), requires_grad = False)
        self.w = nn.Parameter(torch.unsqueeze(torch.unsqueeze(self.w, dim = 0), dim = 0), requires_grad = False)

        t0 = torch.pow(self.q, self.j)
        t1 = torch.pow(1.0 - self.q, self.degree - self.j)

        self.q    = nn.Parameter(self.q, requires_grad = False)
        self.q_pp = nn.Parameter(t0*t1*self.w, requires_grad = False)

    def forward(self, x, p):

        # x: [batch, lead, members]
        # p: [batch, predictors]

        m = x.mean(dim = -1, keepdim = True)
        s = x.std(dim = -1, keepdim = True)

        x = torch.flatten(torch.cat([m, s], dim = -1), start_dim = -2, end_dim = -1)
        x = torch.cat([x, p], dim = -1)

        Y = self.R(x)
       
        return Y.view((Y.shape[0]*self.lead_time, 1, self.degree + 1))

    @jit.ignore
    def loss(self, x, p, f):

        a = self(x, p)
        #a = torch.cumsum(torch.cat([torch.unsqueeze(a[:, :, 0], dim = -1), self.sfp(a[:, :, 1:])], dim = -1), dim = -1)

        # Transform y into latent distribution
        
        lt = f.shape[1]
        bs = f.shape[0]

        q = self.q.expand(bs*lt, -1, -1)
        j = self.j.expand(bs*lt, -1, -1)
        w = self.w.expand(bs*lt, -1, -1)
        q_pp = self.q_pp.expand(bs*lt, -1, -1)

        Q = (q_pp*a).sum(dim = -1)
        f = f.view((bs*lt, 1))

        idx = torch.squeeze(torch.logical_not(torch.isnan(f)))

        dif = f[idx] - Q[idx]

        i0 = dif <  0.0
        i1 = dif >= 0.0

        q = torch.squeeze(q, dim = -1)

        dif[i0] *= q[idx][i0] - 1.0
        dif[i1] *= q[idx][i1]

        return dif.mean(dim = 0).sum()
    
    @jit.export
    def _iF(self, a, f):
        
        bs = f.shape[0]
        lt = f.shape[1]
        nq = f.shape[2]

        f = torch.reshape(f, (bs*lt, nq, 1))

        j = self.j.expand(bs*lt, -1, -1)
        w = self.w.expand(bs*lt, -1, -1)

        q = (torch.pow(f, j)*torch.pow(1.0 - f, self.degree - j)*w*a).sum(dim = -1)

        return q.view((bs, lt, nq))

    @jit.export
    def iF(self, x, p, f):
        
        a = self(x, p)
        #a = torch.cumsum(torch.cat([torch.unsqueeze(a[:, :, 0], dim = -1), self.sfp(a[:, :, 1:])], dim = -1), dim = -1)
        
        return self._iF(a, f)

