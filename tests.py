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
from utilities import global_2_local_id, copy_array_sans_indices
smoke_test = ('CI' in os.environ)
pyro.enable_validation()
pyro.set_rng_seed(0)
from dirichletLearning import *

def bayes_rule_test():  # pass
	b = torch.tensor([.5, .5])
	a_cond_b = torch.tensor([[1., 0.], [.5, .5]])
	b_cond_a = bayes_rule(b, a_cond_b)
	b_cond_a_true = torch.tensor([[2./3., 1./3.], [0., 1.]])
	delta = torch.abs(b_cond_a - b_cond_a_true).sum()
	assert delta < 0.001, delta
	print(b_cond_a)


def rsa_test():  # TODO deal with nans when a word is never used
	# s_0 = torch.tensor([[1,0,0], [1,1,0], [1,1,1]], dtype=torch.float32)
	# l_prior = torch.ones(3) / 3.
	s_0 = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.float32)
	# l_prior = torch.tensor([1/3,2/3])
	print("s_0:\n", s_0)
	s_1 = rsa(s_0, theta=10)
	print("s_1:\n", s_1)
	a = torch.tensor([[0., 2., 5.], [0., 3., 3.]])
	a1 = normalize(a)
	assert torch.abs(torch.ones(a.shape[0]) - a1.sum(dim=1)).sum() < 0.001
	print(a1)


def softmax_test():
	a = torch.tensor([[1., 2.], [1., 5.]])
	b = softmax(a, theta=1)
	b_true = torch.empty(size=a.shape)
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			b_true[i, j] = (torch.exp(a[i, j]))
	b_true = normalize(b_true)
	assert l1_distance(b, b_true) < 0.001, l1_distance(b, b_true)

def context_test():
	a = []
	Context.all_words = a
	a.append('cat')
	assert 'cat' in a.all_words