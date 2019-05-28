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
# torch.set_printoptions(precision=3, sci_mode=False)  #Need to update pytorch to toggle sci_mode
# for i in range(10):
# 	t = dist.Dirichlet(torch.tensor([1000,1000,.0002])).sample()
# 	print(t)

a = torch.tensor([1,2,3,3,3,5])
print(a.dtype)
b = a.bincount()
c = torch.tensor([1.,3.])
print(c)
c.long()
print(c)
c = c.long()
print(c)