import os
import torch
import pyro
import pyro.distributions as dist
from pyro.params.param_store import ParamStoreDict
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

def outer_sum_example():
	#Vectorized!!!
	a0 = torch.arange(3).float().reshape(3,1)
	a1 = torch.arange(3).float().reshape(1,3) * 0.1
	x = a0 + a1
	return x
def subarray_replacement_example():
	x = outer_sum_example()
	b = torch.arange(6).reshape(2,3).float()
	c = [0,2]
	print(x)
	x[c] = b
	print(x)
	x = outer_sum_example()
	c = [[0,0],[1,2]]
	print(x[c])
	# print(x)
def enumerate_example():
	for x,y in enumerate(['a','b']):
		assert type(x) is int
def known_and_believed_example():
	num_items = 2
	num_words = 3
	known_words = torch.zeros((num_items,num_words))
	word_beliefs = torch.ones((num_items,num_words)) * 0.5
	known_words_ids = [[] for i in range(num_items)]
	known_words_ids[0].append(1)
	known_words_ids[0].append(0)

	word_beliefs_adjusted = word_beliefs.clone().detach()
	for item_num in range(num_items):
		word_beliefs_adjusted[item_num][known_words_ids[item_num]] = known_words[item_num][known_words_ids[item_num]]
	print(word_beliefs_adjusted)

# a = torch.zeros(5)
# a[0:2] = 1
# print(a)
def unzip_example():
	a = [['a0','a1'], ['b0', 'b1']]
	b = zip(*a)
	for i in b:
		print(i)

pyro.clear_param_store()
pyro.param("cake", torch.ones(3))
ps = pyro.get_param_store()
print(ps.items())