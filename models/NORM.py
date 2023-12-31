import torch
import torch.nn as nn
import torch.jit as jit
import numpy as np

class Model(nn.Module):

    def __init__(self, lead_time):

        super().__init__()

        self.lead_time = lead_time

        self.sq   = np.sqrt(2.0*np.pi)
        self.sq_r = np.sqrt(2.0) 
        self.sfp = nn.Softplus()

        # Define parameter regression neural network for mean and variance of normal distribution
        self.number_of_outputs = 2*lead_time

    def forward(self):
        return None

    @jit.export
    def set_parameters(self, parameters):

        parameters = parameters.view((parameters.shape[0], self.lead_time, 2))

        self.model_parameters = (parameters[:, :, 0], self.sfp(parameters[:, :, 1]))

    @jit.export
    def pdf(self, f):

        # x: [batch, lead, member]
        # p: [batch, predictors]
        # f: [batch, lead]
   
        mu, sg = self.model_parameters
        
        return torch.exp(-0.5*torch.pow((f - mu)/sg, 2))/(sg*self.sq)

    @jit.export
    def iF(self, f):

        # x: [batch, lead, members]
        # p: [batch, predictors]
        # f: [batch, lead, quantiles]

        mu, sg = self.model_parameters

        mu = mu.view((mu.shape[0], mu.shape[1], 1))
        sg = sg.view((sg.shape[0], sg.shape[1], 1))

        return mu + sg*torch.erfinv(2.0*f - 1.0)*self.sq_r

    @jit.ignore
    def loss(self, f):

        # x: [batch, lead, member]
        # p: [batch, predictors]
        # f: [batch, lead]
        idx = torch.logical_not(torch.isnan(f))

        if idx.sum() == 0:
            return None

        mu, sg = self.model_parameters

        mu = mu[idx]
        sg = torch.clamp(sg[idx], min = 1e-6)
        f = f[idx]

        T0 = torch.log(sg)
        T1 = 0.5*torch.pow((mu - f)/sg, 2)

        L = T0 + T1
        return L.mean()

