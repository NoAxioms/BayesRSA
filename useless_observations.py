import os, copy
import time
from collections import namedtuple
import warnings
warnings.simplefilter('always')
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from utilities import *
smoke_test = ('CI' in os.environ)
pyro.enable_validation()
pyro.set_rng_seed(0)

def model(observation):
	latent = pyro.sample('latent', dist.Categorical(torch.ones(3)/3))
	# print(latent)
	observable = pyro.sample('observable', dist.Categorical(torch.ones(3)/3), obs = observation)

def guide(observation):
	belief = pyro.param('belief', torch.ones(3)/3, constraint = constraints.simplex)
	latent = pyro.sample('latent', dist.Categorical(belief))

if __name__ == "__main__":
	pyro.clear_param_store()
	adam_params = {"lr": 0.05, "betas": (0.95, 0.999)}
	optimizer = Adam(adam_params)
	svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
	# gesture = torch.tensor([2.3562, 0.0000, arm_length])
	for s in range(10000):
		svi.step(torch.tensor(0))
		if s % 100 == 0:
			print(pyro.param('belief'))