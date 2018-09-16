import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def evan_mask(d_in, d_hid):
  mask_in = np.ones((d_hid, d_in))
  degree_cnts = np.diff(np.concatenate([[0], np.sort(np.random.choice(np.arange(d_hid-1)+1, d_in-2, replace=False)), [d_hid]]))
  cnt = 0
  for i_col, deg in enumerate(degree_cnts):
    cnt += deg
    mask_in[:cnt, i_col+1] = 0

  mask_hid = np.ones((d_hid, d_hid))
  degree_cnts2 = np.diff(np.concatenate([[0], np.sort(np.random.choice(np.arange(d_hid-1)+1, d_in-2, replace=False)), [d_hid]]))

  cnt1, cnt2 = 0, 0
  for deg1, deg2 in zip(degree_cnts, degree_cnts2):
    
    mask_hid[:cnt2, cnt1:(cnt1+deg1)] = 0.0
    cnt1 += deg1
    cnt2 += deg2

  mask_out = np.zeros((d_in, d_hid))
  cnt = 0
  for i_row, deg in enumerate(degree_cnts2):
    cnt += deg
    mask_out[i_row+1, :cnt] = 1

  prod = np.dot(np.dot(mask_out, mask_hid), mask_in)
  prod[prod > 1] = 1
  assert np.all(np.sum(prod, axis=1) == np.arange(d_in)), 'AR property violation'
  mask_out = np.concatenate([mask_out, mask_out], axis=0)

  to_float = lambda lst: list(map(lambda x: x.astype(np.float32), lst))
  to_torch = lambda lst: list(map(lambda x: torch.from_numpy(x), lst))
  return to_torch(to_float([mask_in, mask_hid, mask_out]))
  
    

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output
    
    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)

    def forward(self, inputs):
        return F.linear(inputs, self.weight * self.mask, self.bias)


nn.MaskedLinear = MaskedLinear


class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self, num_inputs, num_hidden):
        super(MADE, self).__init__()
  
        input_mask, hidden_mask, output_mask = evan_mask(num_inputs, num_hidden)
        self.register_buffer('input_mask', input_mask)
        self.register_buffer('hidden_mask', hidden_mask)
        self.register_buffer('output_mask', output_mask)

        #input_mask = get_mask(
        #    num_inputs, num_hidden, num_inputs, mask_type='input')
        #hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        #output_mask = get_mask(
        #    num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.main = nn.Sequential(
            nn.MaskedLinear(num_inputs, num_hidden, input_mask), nn.LeakyReLU(),
            nn.BatchNorm1d(num_hidden),
            nn.MaskedLinear(num_hidden, num_hidden, hidden_mask), nn.LeakyReLU(),
            nn.BatchNorm1d(num_hidden),
            nn.MaskedLinear(num_hidden, 2*num_inputs, output_mask))

    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            m, a = self.main(inputs).chunk(2, 1)
            a = torch.tanh(a)
                    
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                m, a = self.main(x).chunk(2, 1)
                a = torch.tanh(a)
                x[:, i_col] = inputs[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col] 
                
            return x, -a.sum(-1, keepdim=True)
                


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)


class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs, mode='direct'):
        if self.initialized == False:
            self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        if mode == 'direct':
            return (
                inputs - self.bias) * torch.exp(self.weight), self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs * torch.exp(
                -self.weight) + self.bias, -self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)


class InvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(InvertibleMM, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_inputs, num_inputs))
        nn.init.orthogonal_(self.W)

    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            return inputs @ self.W, torch.log(torch.abs(torch.det(
                self.W))).unsqueeze(0).unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(self.W), -torch.log(
                torch.abs(torch.det(self.W))).unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)


class HandOrder(nn.Module):
    """Prepare the data to be in the desired joint order
    """

    def __init__(self, num_inputs):
        super(HandOrder, self).__init__()
        
        self.perm = (np.array(
          [0, 5, 1, 9, 13, 17, 6, 2, 10, 14, 18, 7, 3, 11, 15, 19, 8, 4, 12, 16, 20])[:,None] + np.arange(3)[None,:]).flatten()
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class Shuffle(nn.Module):
    """ An implementation of a shuffling layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Shuffle, self).__init__()
        perm = np.random.permutation(num_inputs)
        inv_perm = np.argsort(perm)

        self.perm = torch.from_numpy(perm)
        self.inv_perm = torch.from_numpy(inv_perm)
        self.register_buffer('perm_vec', self.perm)
        self.register_buffer('inv_perm_vec', self.inv_perm)

    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, num_hidden=64):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs

        self.main = nn.Sequential(
            nn.Linear(num_inputs // 2, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, 2 * (self.num_inputs - num_inputs // 2)))

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            x_a, x_b = inputs.chunk(2, dim=-1)
            log_s, t = self.main(x_b).chunk(2, dim=-1)
            s = torch.exp(log_s)

            y_a = x_a * s + t
            y_b = x_b
            return torch.cat([y_a, y_b], dim=-1), log_s.sum(-1, keepdim=True)
        else:
            y_a, y_b = inputs.chunk(2, dim=-1)
            log_s, t = self.main(y_b).chunk(2, dim=-1)
            s = torch.exp(-log_s)
            x_a = (y_a - t) * s
            x_b = y_b
            return torch.cat([x_a, x_b], dim=-1), -log_s.sum(-1, keepdim=True)


def log_wrapped_normal_pdf(angle, mean, std, k_max=50):
    pdf = torch.zeros_like(angle).to(angle.device)
    for k in range(-k_max, k_max+1):
        pdf += torch.exp(-(angle - mean + 2*np.pi*k)**2 / (2*std)**2)

    return torch.log(pdf/(std*np.sqrt(2*np.pi)))


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """
    def __init__(self, num_blocks, num_inputs, num_hidden, device):
    
        modules = []
        for i_block in range(num_blocks):
          if i_block == num_blocks - 1:
            modules += [
              MADE(num_inputs, num_hidden),
              Shuffle(num_inputs)]
          else:
            modules += [
              MADE(num_inputs, num_hidden),
              #BatchNormFlow(num_inputs),
              Shuffle(num_inputs)]

        super(FlowSequential, self).__init__(*modules)

        self.prior = torch.distributions.MultivariateNormal(torch.zeros(num_inputs-2).to(device), torch.eye(num_inputs-2).to(device))

    def f(self, x, azim, elev, wn_std=0.5):
        """wn_std: was 0.5"""
        z, log_det = self.forward(x, mode='direct')

        log1 = log_wrapped_normal_pdf(z[:,0], azim.view(-1), wn_std) 
        log2 = log_wrapped_normal_pdf(z[:,1], elev.view(-1), wn_std)
        
        log_prob = log1 + \
                   log2 + \
                   self.prior.log_prob(z[:,2:]) + \
                   log_det.view(-1)
        return z, log_prob
    
    def g(self, z):
        x, _ = self.forward(z, mode='inverse')
        return x
 
    def log_prob(self, x, azim, elev):
        _, log_prob = self.f(x, azim, elev)
        return log_prob 

    def forward(self, inputs, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']

        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, mode)
                logdets += logdet

        return inputs, logdets


if __name__ == '__main__':
  inn, hid, out = evan_mask(63, 100)
 
