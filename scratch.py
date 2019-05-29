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

#Vectorized!!!
a0 = torch.arange(3).float().reshape(3,1)
a1 = torch.arange(3).float().reshape(1,3) * 0.1
b = torch.arange(6).reshape(2,3).float()
x = a0 + a1
c = [0,2]
x[c] = b
for x,y in enumerate(['a','b']):
	assert type(x) is int

print([list(range(3))])