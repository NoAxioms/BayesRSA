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

def cart2sph(cart):
	r = torch.norm(cart,2)
	az = torch.atan2(cart[1],cart[0])
	hxy = torch.norm(cart[0:2],2)
	el = torch.atan2(cart[2],hxy)
	return torch.tensor([az,el,r])

def sph2cart(sph):
	rcos_theta = sph[2] * torch.cos(sph[1])
	x = rcos_theta * torch.cos(sph[0])
	y = rcos_theta * torch.sin(sph[0])
	z = sph[2] * torch.sin(1)
	return torch.tensor([x,y,z])

def gesture_model(item_locs, head_loc, obs_gesture, arm_length = 0.5, noise = .1):
	"""
	:param target: target_location - human_location
	"""
	target_item = pyro.sample('target_item', dist.Categorical(pyro.param('item_probs')))
	target_loc = item_locs[target_item]
	ideal_vector = cart2sph(target_loc - head_loc)
	dist2target = ideal_vector[2]
	# ideal_vector[2] = arm_length
	covariance_matrix = torch.eye(3)
	covariance_matrix[2] = 0.001
	covariance_matrix *= noise * (dist2target - arm_length)  

	distance = torch.norm(ideal_vector,2)
	gesture = pyro.sample('gesture', dist.MultivariateNormal(loc=ideal_vector, covariance_matrix = torch.eye(3) * noise * distance), obs = obs_gesture)
	return gesture

def gesture_guide(item_locs, head_loc, obs_gesture, arm_length = 0.5, noise = .1):
	target_item = pyro.sample('target_item', dist.Categorical(pyro.param('item_probs')))



# for i in item_locs:
# 	j = i / torch.norm(i,2)
# 	print(cart2sph(j))
if __name__ == "__main__":
	pyro.clear_param_store()
	item_locs = [[-1.,1,0], [0.,1,0], [1.,1,0]]
	item_locs = [torch.tensor(i) for i in item_locs]
	head_loc=torch.tensor([0,0,0.])
	arm_length = 0.5
	num_items = len(item_locs)
	item_probs = pyro.param('item_probs', torch.ones(num_items)/num_items, constraint=constraints.simplex)
	adam_params = {"lr": 0.05, "betas": (0.95, 0.999)}
	optimizer = Adam(adam_params)
	svi = SVI(gesture_model, gesture_guide, optimizer, loss=Trace_ELBO())
	gesture = torch.tensor([2.3562, 0.0000, arm_length])
	for s in range(1000):
		svi.step(item_locs,head_loc, gesture, arm_length, 0.0001)
		if s % 100 == 0:
			print(pyro.param('item_probs'))