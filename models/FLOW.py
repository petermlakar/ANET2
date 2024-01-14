import torch
import torch.nn as nn
import torch.jit as jit
import numpy as np

class SplineBlock(nn.Module):

    def  __init__(self):
        super().__init__()

    def forward(self, x):

        # x:    [batch*lead, 1, 1]
        # prm_: [batch*lead, 1, nknots]

        i = (x > self.t).sum(dim = -1)
        k0 = i - 1
        k1 = i

        eq0 = i == 0
        eq1 = i == self.t.shape[-1]

        k0[eq0] = 0
        k1[eq0] = 1

        k0[eq1] = self.t.shape[-1] - 2
        k1[eq1] = self.t.shape[-1] - 1

        idx = torch.logical_not(torch.logical_or(eq0, eq1))

        k0 = torch.unsqueeze(k0, dim = 1)
        k1 = torch.unsqueeze(k1, dim = 1)

        t0 = torch.squeeze(torch.gather(self.t, -1, k0), dim = 1)
        t1 = torch.squeeze(torch.gather(self.t, -1, k1), dim = 1)

        y0 = torch.squeeze(torch.gather(self.y, -1, k0), dim = 1)
        y1 = torch.squeeze(torch.gather(self.y, -1, k1), dim = 1)

        d0 = torch.squeeze(torch.gather(self.d, -1, k0), dim = 1)
        d1 = torch.squeeze(torch.gather(self.d, -1, k1), dim = 1)

        _t0 = t0[idx]
        _t1 = t1[idx]

        _y0 = y0[idx]
        _y1 = y1[idx]

        _d0 = d0[idx]
        _d1 = d1[idx]

        delta_t = _t1 - _t0
        delta_y = _y1 - _y0

        s = delta_y/delta_t

        x = torch.squeeze(x, dim = -1)
        e = (x[idx] - _t0)/delta_t

        n0 = delta_y*(s*e*e + _d0*e*(1.0 - e))
        n1 = s + (_d1 + _d0 - 2.0*s)*e*(1.0 - e)

        p = torch.clone(x)

        if idx.sum() > 0:
            p[idx] = _y0 + n0/n1

        # Compute asimptotics

        a0 = d0
        b0 = y0 - a0*t0

        a1 = d1
        b1 = y1 - a1*t1

        if eq0.sum() > 0:
            p[eq0] = a0[eq0]*x[eq0] + b0[eq0]
        if eq1.sum() > 0:
            p[eq1] = a1[eq1]*x[eq1] + b1[eq1]

        return torch.unsqueeze(p, dim = -1)

    @jit.export
    def backward(self, y):

        # y:    [batch*lead, quantiles, 1]
        # prm_: [batch*lead, 1, nknots]

        i = (y > self.y).sum(dim = -1)
        k0 = i - 1
        k1 = i

        eq0 = i == 0
        eq1 = i == self.y.shape[-1]

        k0[eq0] = 0
        k1[eq0] = 1

        k0[eq1] = self.y.shape[-1] - 2
        k1[eq1] = self.y.shape[-1] - 1

        idx = torch.logical_not(torch.logical_or(eq0, eq1))

        k0 = torch.unsqueeze(k0, dim = 1)
        k1 = torch.unsqueeze(k1, dim = 1)

        t0 = torch.squeeze(torch.gather(self.t, -1, k0), dim = 1)
        t1 = torch.squeeze(torch.gather(self.t, -1, k1), dim = 1)

        y0 = torch.squeeze(torch.gather(self.y, -1, k0), dim = 1)
        y1 = torch.squeeze(torch.gather(self.y, -1, k1), dim = 1)

        d0 = torch.squeeze(torch.gather(self.d, -1, k0), dim = 1)
        d1 = torch.squeeze(torch.gather(self.d, -1, k1), dim = 1)

        _t0 = t0[idx]
        _t1 = t1[idx]

        _y0 = y0[idx]
        _y1 = y1[idx]

        _d0 = d0[idx]
        _d1 = d1[idx]

        delta_t = _t1 - _t0
        delta_y = _y1 - _y0

        s = delta_y/delta_t
        o = _d1 + _d0 - 2.0*s

        y = torch.squeeze(y, dim = -1)

        df = y[idx] - _y0
        a = delta_y*(s - _d0) + df*o
        b = delta_y*_d0 - df*o
        c = -s*df

        if idx.sum() > 0:
            y[idx] = 2.0*c*delta_t/(-b - torch.sqrt(b*b - 4.0*a*c)) + _t0

        # Compute asimptotics

        a0 = d0
        b0 = y0 - a0*t0

        a1 = d1
        b1 = y1 - a1*t1

        if eq0.sum() > 0:
            y[eq0] = (y[eq0] - b0[eq0])/a0[eq0]
        if eq1.sum() > 0:
            y[eq1] = (y[eq1] - b1[eq1])/a1[eq1]

        return torch.unsqueeze(y, dim = -1)

    @jit.export
    def dt(self, x):

        # x:    [batch*lead, 1, 1]
        # prm_: [batch*lead, 1, nknots]
       
        i = (x > self.t).sum(dim = -1)
        k0 = i - 1
        k1 = i

        eq0 = i == 0
        eq1 = i == self.t.shape[-1]

        k0[eq0] = 0
        k1[eq0] = 1

        k0[eq1] = self.t.shape[-1] - 2
        k1[eq1] = self.t.shape[-1] - 1

        idx = torch.logical_not(torch.logical_or(eq0, eq1))

        k0 = torch.unsqueeze(k0, dim = 1)
        k1 = torch.unsqueeze(k1, dim = 1)

        t0 = torch.squeeze(torch.gather(self.t, -1, k0), dim = 1)
        t1 = torch.squeeze(torch.gather(self.t, -1, k1), dim = 1)

        y0 = torch.squeeze(torch.gather(self.y, -1, k0), dim = 1)
        y1 = torch.squeeze(torch.gather(self.y, -1, k1), dim = 1)

        d0 = torch.squeeze(torch.gather(self.d, -1, k0), dim = 1)
        d1 = torch.squeeze(torch.gather(self.d, -1, k1), dim = 1)

        _t0 = t0[idx]
        _t1 = t1[idx]

        _y0 = y0[idx]
        _y1 = y1[idx]

        _d0 = d0[idx]
        _d1 = d1[idx]

        delta_t = _t1 - _t0
        delta_y = _y1 - _y0
        s = delta_y/delta_t

        x = torch.squeeze(x, dim = -1)
        e = (x[idx] - _t0)/delta_t

        a0 = delta_y*s
        a1 = delta_y*_d0
        b = _d1 + _d0 - 2.0*s
        
        p = torch.clone(x)

        if idx.sum() > 0:
            
            div0 = torch.pow(s, 2)*(_d1*torch.pow(e, 2) + 2.0*s*e*(1.0 - e) + _d0*torch.pow(1.0 - e, 2))
            div1 = torch.pow(s + b*e*(1.0 - e), 2)

            p[idx] = div0/div1

        if eq0.sum() > 0:
            p[eq0] = d0[eq0]

        if eq1.sum() > 0:
            p[eq1] = d1[eq1]

        return torch.unsqueeze(p, dim = -1)

    @jit.export
    def set(self, t, y, d):

        self.t = t
        self.y = y
        self.d = d

class Model(nn.Module):

    def __init__(self, lead_time, nblocks = 1, nknots = 6):

        super().__init__()

        self.lead_time = lead_time

        self.sqrt2   = torch.sqrt(torch.tensor(2.0, dtype = torch.float32))
        self.sqrt2pi = torch.sqrt(torch.tensor(2.0, dtype = torch.float32)*np.pi)
        self.sfp = nn.Softplus()
    
        self.nblocks = nblocks
        self.nknots  = nknots

        self.number_of_parameters = self.nblocks*self.nknots*2
       
        self.spline_block = SplineBlock()

        # Define parameter regression neural network for estimating the normalizing spline flow parameters (knot-value pairs)
        self.number_of_outputs = self.number_of_parameters*self.lead_time 

        self.lq = nn.Parameter(torch.unsqueeze(torch.unsqueeze(torch.arange(1.0/52.0, 1.0, step = 1.0/52.0), dim = 0), dim = 0), requires_grad = False)
        self.ones = nn.Parameter(torch.ones((1, 1), dtype = torch.float32), requires_grad = False)

    def forward(self):
        return None

    @jit.export
    def set_parameters(self, parameters):

        parameters = parameters.view((parameters.shape[0], self.lead_time, self.nblocks, 2, self.nknots))

        self.model_parameters = (torch.flatten(parameters[:, :, :, 0], start_dim = 0, end_dim = 1), torch.flatten(parameters[:, :, :, 1], start_dim = 0, end_dim = 1))

    @jit.export
    def pdf(self, f):
        
        fs0 = f.shape[0]
        fs1 = f.shape[1]

        f = f.view((f.shape[0]*f.shape[1], 1, 1))
        prm_t, prm_y = self.model_parameters

        dt = []

        for i in torch.arange(self.nblocks):

            t = torch.cumsum(torch.cat([prm_t[:, i, 0][..., None], 1e-3 + self.sfp(prm_t[:, i, 1:])], dim = -1), dim = -1)
            y = torch.cumsum(torch.cat([prm_y[:, i, 0][..., None], 1e-3 + self.sfp(prm_y[:, i, 1:])], dim = -1), dim = -1)

            h  = t[:, 1:] - t[:, :-1]
            df = (y[:, 1:] - y[:, :-1])/h

            di = df[:, :-1]*df[:, 1:]/((y[:, 2:] - y[:, :-2])/(t[:, 2:] - t[:, :-2]))
            d = torch.cat([self.ones.expand(di.shape[0], -1), di, self.ones.expand(di.shape[0], -1)], dim = -1)

            self.spline_block.set(t[:, None], y[:, None], d[:, None])
            dt.append(self.spline_block.dt(f))
            f  = self.spline_block(f)

        f  = torch.squeeze(f)
        dt = torch.squeeze(torch.stack(dt, dim = 0).prod(dim = 0))

        f  = f.view((fs0, fs1))
        dt = dt.view((fs0, fs1)) 

        return torch.exp(-0.5*torch.pow(f, 2))*dt/self.sqrt2pi

    @jit.export
    def nloglikelihood(self, f):

        # x: [batch, lead, member]
        # p: [batch, predictors]
        # f: [batch, lead]
        fs0 = f.shape[0]
        fs1 = f.shape[1]

        f = f.view((f.shape[0]*f.shape[1], 1, 1))
        prm_t, prm_y = self.model_parameters

        dt = []

        idx = torch.isnan(f)

        idx = torch.logical_not(idx).view((fs0, fs1))

        for i in torch.arange(self.nblocks):

            t = torch.cumsum(torch.cat([torch.unsqueeze(prm_t[:, i, 0], dim = -1), 1e-3 + self.sfp(prm_t[:, i, 1:])], dim = -1), dim = -1)
            y = torch.cumsum(torch.cat([torch.unsqueeze(prm_y[:, i, 0], dim = -1), 1e-3 + self.sfp(prm_y[:, i, 1:])], dim = -1), dim = -1)

            h  = t[:, 1:] - t[:, :-1]
            df = (y[:, 1:] - y[:, :-1])/h

            di = df[:, :-1]*df[:, 1:]/((y[:, 2:] - y[:, :-2])/(t[:, 2:] - t[:, :-2]))
            d1 = torch.unsqueeze(torch.pow(df[:, 0], 2)/((y[:, 2] - y[:, 0])/(t[:, 2] - t[:, 0])), dim = -1)
            dn = torch.unsqueeze(torch.pow(df[:, -1], 2)/((y[:, -1] - y[:, -3])/(t[:, -1] - t[:, -3])), dim = -1)

            d = torch.cat([d1, di, dn], dim = -1)

            self.spline_block.set(torch.unsqueeze(t, dim = 1), torch.unsqueeze(y, dim = 1), torch.unsqueeze(d, dim = 1))

            block_dt = self.spline_block.dt(f)
            log_dt = torch.log(block_dt)

            dt.append(log_dt)
            f  = self.spline_block(f)

        f  = torch.squeeze(f)
        dt = torch.squeeze(torch.stack(dt, dim = 0).sum(dim = 0))

        f  = f.view((fs0, fs1))
        dt = dt.view((fs0, fs1)) 

        return torch.pow(f, 2)*0.5 - dt

    @jit.export
    def iF(self, f):

        # x: [batch, lead, members]
        # p: [batch, predictors]
        # f: [batch, lead, quantiles]

        prm_t, prm_y = self.model_parameters
        invf = torch.erfinv(2.0*f - 1.0)*self.sqrt2

        bs = f.shape[0]
        lt = self.lead_time
        nq = f.shape[-1]

        return self.backward(prm_t, prm_y, invf.view((bs*lt, nq, 1))).view((bs, lt, nq))

    @jit.export
    def backward(self, prm_t, prm_y, p):

        # p: [batch*lead, quantiles, 1]
        # prm_: [batch*lead, nsplines, nknots]

        for i in torch.flip(torch.arange(self.nblocks), dims = (0,)):

            t = torch.cumsum(torch.cat([prm_t[:, i, 0][..., None], 1e-3 + self.sfp(prm_t[:, i, 1:])], dim = -1), dim = -1)
            y = torch.cumsum(torch.cat([prm_y[:, i, 0][..., None], 1e-3 + self.sfp(prm_y[:, i, 1:])], dim = -1), dim = -1)

            h  = t[:, 1:] - t[:, :-1]
            df = (y[:, 1:] - y[:, :-1])/h

            di = df[:, :-1]*df[:, 1:]/((y[:, 2:] - y[:, :-2])/(t[:, 2:] - t[:, :-2]))
            d = torch.cat([self.ones.expand(di.shape[0], -1), di, self.ones.expand(di.shape[0], -1)], dim = -1)

            self.spline_block.set(t[:, None], y[:, None], d[:, None])
            p = self.spline_block.backward(p)

        return p

    @jit.ignore
    def loss(self, f):

        # x: [batch, lead, member]
        # p: [batch, predictors]
        # f: [batch, lead]
        f = f.view((f.shape[0]*f.shape[1], 1, 1))
        idx = torch.squeeze(torch.logical_not(torch.isnan(f)))

        if idx.sum() == 0:
            return None

        prm_t, prm_y = self.model_parameters

        # Transform y into latent distribution

        f = f[idx]

        dt = 0.0

        for i in range(self.nblocks):

            t = torch.cumsum(torch.cat([prm_t[:, i, 0][..., None], 1e-3 + self.sfp(prm_t[:, i, 1:])], dim = -1), dim = -1)
            y = torch.cumsum(torch.cat([prm_y[:, i, 0][..., None], 1e-3 + self.sfp(prm_y[:, i, 1:])], dim = -1), dim = -1)

            h  = t[:, 1:] - t[:, :-1]
            df = (y[:, 1:] - y[:, :-1])/h

            di = df[:, :-1]*df[:, 1:]/((y[:, 2:] - y[:, :-2])/(t[:, 2:] - t[:, :-2]))
            d = torch.cat([self.ones.expand(di.shape[0], -1), di, self.ones.expand(di.shape[0], -1)], dim = -1)

            t = t[idx]
            y = y[idx]
            d = d[idx]

            self.spline_block.set(t[:, None], y[:, None], d[:, None])
            block_dt = self.spline_block.dt(f)
            log_dt = torch.log(block_dt)

            dt += log_dt
            f  = self.spline_block(f)

        f  = torch.squeeze(f)
        dt = torch.squeeze(dt)

        return (torch.pow(f, 2)*0.5 - dt).mean()


