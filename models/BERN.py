import torch
import torch.nn as nn
import torch.jit as jit

import numpy as np

import math

class Model(nn.Module):

    def __init__(self, lead_time, positive_support = False, monotone = False):

        super().__init__()

        self.lead_time = lead_time
        self.sfp = nn.Softplus()

        self.degree = 12
        self.nquantiles = 100

        ########################################################################
        # Enforce a positive support for the Bernstein quanilte function
        # and enforce the monotonicity of the coefficients which in turn
        # results in the monotonicity of the quantile function.
        self.positive_support = positive_support
        self.monotone = monotone

        # Bernstein quantile function coefficient regression neural network
        self.number_of_outputs = (self.degree + 1)*self.lead_time

        ########################################################################
        # Compute binomial coefficients and the remaining required parameters
        
        step = 1.0/(self.nquantiles + 1)
        self.q = torch.arange(step, 1.0, step = step)

        self.w = torch.ones(self.degree + 1, dtype = torch.float32)
        for i in torch.arange(0, self.degree + 1, dtype = torch.int32):
            self.w[i] = math.comb(self.degree, i)
        self.j = torch.arange(0, self.degree + 1, dtype = torch.float32)

        self.q = self.q.view((1, self.nquantiles, 1))

        t0 = torch.pow(self.q, self.j)
        t1 = torch.pow(1.0 - self.q, self.degree - self.j)

        ########################################################################
        # Define as parameters such that they will be moved to gpu when .to is 
        # invoked on the module.

        self.j = nn.Parameter(self.j[None, None], requires_grad = False)
        self.w = nn.Parameter(self.w[None, None], requires_grad = False)

        self.q    = nn.Parameter(self.q, requires_grad = False)
        self.q_pp = nn.Parameter(t0*t1*self.w, requires_grad = False)

    def forward(self):
        return None

    @jit.export
    def set_parameters(self, parameters):
        self.model_parameters = parameters.view((parameters.shape[0]*self.lead_time, 1, self.degree + 1))
   
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
    def iF(self, f):
        
        a = self.model_parameters

        # Enforce monotonicity by means of cumulative sum of positive increments
        if self.monotone:

            if self.positive_support:
                a[:, :, 0] = self.sfp(a[:, :, 0])

            a = torch.cumsum(torch.cat([a[:, :, 0][..., None], self.sfp(a[:, :, 1:])], dim = -1), dim = -1)

        elif self.positive_support:
            a = self.sfp(a)

        return self._iF(a, f)

    @jit.ignore
    def loss(self, f):

        a = self.model_parameters

        # Enforce monotonicity by means of cumulative sum of positive increments
        if self.monotone:
            
            if self.positive_support:
                a[:, :, 0] = self.sfp(a[:, :, 0])

            a = torch.cumsum(torch.cat([a[:, :, 0][..., None], self.sfp(a[:, :, 1:])], dim = -1), dim = -1)

        elif self.positive_support:
            a = self.sfp(a)

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
 
