import os
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.optim import Adam


# from rsaClass import RSA
# from utilities import softmax
smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('0.3.1')
pyro.enable_validation()
pyro.set_rng_seed(0)

# num_items = 5
# vocab_size = 2
# concentrations = 0.5 * torch.ones(num_items,vocab_size)
# t = pyro.sample("t", dist.Dirichlet(concentrations))
# # x = pyro.sample("x", dist.Categorical(t[0]))
# print(concentrations)
# print(t)
# print(0.5 * torch.ones(5))
a = torch.tensor([[0] * 5])
b = torch.tensor([[1] * 5])
c = torch.tensor([[2] * 5])
z = torch.cat((a,b,c),0)
print(torch.tensor([[0] * 5, [1] * 5, [2] * 5]))