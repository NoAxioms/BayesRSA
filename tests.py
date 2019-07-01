import os, copy, timeit, functools
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
from main import *

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

def cart2sph_test():
	#broadcasting is faster (or just comparable?) than legacy when we have at least 5-7 items.
	cart = torch.tensor([[1.,0,0], [2,3,4],[2,3,4],[2,3,4],[2,3,4],[2,3,4],[2,3,4]])
	sphere_legacy = torch.empty(cart.shape[0],3)
	for row in range(cart.shape[0]):	
		sphere_legacy[row] = cart2sph_legacy(cart[row])
	sphere_broadcasting = cart2sph(cart)
	# print(l1_distance(sphere_broadcasting,sphere_legacy))
	assert l1_distance(sphere_broadcasting,sphere_legacy) == 0, "{}\n{}".format(sphere_broadcasting,sphere_legacy)
	sphere_1 = cart2sph(cart[1])
	# print(l1_distance(sphere_1,sphere_broadcasting[1]))
	assert l1_distance(sphere_1,sphere_broadcasting[1]) == 0, "{}\n{}".format(sphere_1,sphere_broadcasting[1])
	#Time tests
	legacy_timer_half = timeit.Timer(functools.partial(cart2sph_legacy,cart[1]))
	legacy_time = legacy_timer_half.timeit(1000) * cart.shape[0]
	print("legacy cart2sph time: {}".format(legacy_time))
	broadcasting_timer = timeit.Timer(functools.partial(cart2sph,cart))
	broadcasting_time = broadcasting_timer.timeit(1000)
	print("broadcasting cart2sph time: {}".format(broadcasting_time))


def sph2cart_test():
	#broadcasting 3x faster for 5 items
	sph = torch.tensor([[0,0,1],[.2,.3,4], [.2,.3,4], [.2,.3,4], [.2,.3,4]])
	cart_legacy = torch.empty(sph.shape[0],3)
	for row in range(sph.shape[0]):
		cart_legacy[row] = sph2cart_legacy(sph[row])
	cart_broadcasting = sph2cart(sph)
	assert l1_distance(cart_broadcasting,cart_legacy) == 0, "\n{}\n{}".format(cart_broadcasting,cart_legacy)
	cart_1 = sph2cart(sph[1])
	assert l1_distance(cart_1, cart_broadcasting[1]) == 0
	#Time tests
	legacy_timer_half = timeit.Timer(functools.partial(sph2cart_legacy,sph[1]))
	legacy_time = legacy_timer_half.timeit(1000) * sph.shape[0]
	print("legacy sph2cart time: {}".format(legacy_time))
	broadcasting_timer = timeit.Timer(functools.partial(sph2cart,sph))
	broadcasting_time = broadcasting_timer.timeit(1000)
	print("broadcasting sph2cart time: {}".format(broadcasting_time))

def tensor_view_test():
	a = torch.tensor([[[0,1,2],[10,11,12]],[[100,101,102],[110,111,112]]])
	b = a.view(-1,3)
	c = b.view(a.shape)
	print(b)
	print(l1_distance(a,c))


def att_set_test():
	#Ripped from main.py. Probably won't work.
	num_items = 3
	initialize_knowledge(num_items)
	# num_words = 0
	all_words = ['a','b','c','d']
	Context.all_words = all_words
	context = Context(items=(0,1,2))
	context_list = [context]
	pyro.clear_param_store()
	adam_params = {"lr": 0.05, "betas": (0.95, 0.999)}
	optimizer = Adam(adam_params)
	svi = SVI(language_model, language_guide, optimizer, loss=Trace_ELBO())
	revelations = []
	revelations.append(Revelation(0,[0],[1]))
	revelations.append(Revelation(1,[0],[1]))
	svi_args = {
		'contexts':context_list,
		'num_items':num_items,
		'use_rsa':True
	}
	time_limit = 1
	context.hear(0,'a')
	update_with_revelations(svi=svi, revelations=[revelations[0]],svi_args=svi_args, time_limit=time_limit)
	context.hear(1,'b')
	update_with_revelations(svi=svi, revelations=[revelations[1]],svi_args=svi_args, time_limit=time_limit)
	print(language_model(context_list, num_items))

def enumeration_time_test():
	"""
	Compares speed of inference using enumeration v not using enumeration
	"""
	num_trials = 2
	"""
	The order of running seems to affect outcome - the second is faster, especially when the second is enumeration. Problem with globals?
	Possibilities:
	global/param not being reset
	Amortization under the hood
	
	Checks:
	Run run_trials twice, with different gestures/vocab/desired items, see if second run is faster and accurate.
	To facilitate this, refactor run_trials to accept appropriate arguments.
	"""
	item_locs0 = torch.tensor([[-1.,1,0], [0.,1,0], [1.,1,0]])
	autopilot_utterances0 = [[['face']] * 100] + [[['moustache']] * 100]
	autopilot_gestures0 = [[torch.tensor([0,0,0,-1,1,0], dtype=torch.float)] * 100]  + [[torch.tensor([0,0,0,0,1,0], dtype=torch.float)] * 100]
	autopilot_targets0 = [0,1]

	item_locs1 = torch.tensor([[-6.,1,0], [-1.,1,0], [3.,1,0]])
	autopilot_utterances1 = [[['face']] * 100] + [[['moustache']] * 100]
	autopilot_gestures1 = [[torch.tensor([0,0,0,3,1,0], dtype=torch.float)] * 100]  + [[torch.tensor([0,0,0,-6,1,0], dtype=torch.float)] * 100]
	autopilot_targets1 = [2,0]
	domain_args0 = {
		"item_locs":item_locs0,
		"autopilot_utterances": autopilot_utterances0,
		"autopilot_gestures": autopilot_gestures0,
		"autopilot_targets": autopilot_targets0,
		"autopilot":True
	}
	domain_args1 = {
		"item_locs":item_locs1,
		"autopilot_utterances": autopilot_utterances1,
		"autopilot_gestures": autopilot_gestures1,
		"autopilot_targets": autopilot_targets1,
		"autopilot":True
	}
	enumeration_beliefs = run_trials(use_enumeration = True, num_trials = num_trials, verbose=False, stop_method = "convergence", domain_args = domain_args0)
	sans_enumeration_beliefs = run_trials(use_enumeration=False, num_trials=num_trials, verbose=False, stop_method = "convergence", domain_args = domain_args0)

	# enumeration_beliefs = run_trials(use_enumeration = True, num_trials = num_trials)
	# sans_enumeration_beliefs = run_trials(use_enumeration=False, num_trials=num_trials)

	for t in range(num_trials):
		print("Enumeration beliefs for trial {}. Num_steps: {}".format(t,len(enumeration_beliefs[t])))
		for step_num, d in enumerate(enumeration_beliefs[t]):
			# print(d["target_item"])
			print(d["svi_time"])
			print(d["lexicon"])
		print("sans enumeration beliefs for trial {}. Num_steps: {}".format(t,len(sans_enumeration_beliefs[t])))

		for step_num, d in enumerate(sans_enumeration_beliefs[t]):
			print(d["svi_time"])
			print(d["lexicon"])

def run_trials_independence_test():
	"""
	Tests whether sequential calls to run_trials are independent
	THE SECOND RUN IS FASTER?
	"""
	item_locs0 = torch.tensor([[-1.,1,0], [0.,1,0], [1.,1,0]])
	autopilot_utterances0 = [[['face']] * 100] + [[['moustache']] * 100]
	autopilot_gestures0 = [[torch.tensor([0,0,0,-1,1,0], dtype=torch.float)] * 100]  + [[torch.tensor([0,0,0,0,1,0], dtype=torch.float)] * 100]
	autopilot_targets0 = [0,1]

	item_locs1 = torch.tensor([[-6.,1,0], [-1.,1,0], [3.,1,0]])
	autopilot_utterances1 = [[['face']] * 100] + [[['moustache']] * 100]
	autopilot_gestures1 = [[torch.tensor([0,0,0,3,1,0], dtype=torch.float)] * 100]  + [[torch.tensor([0,0,0,-6,1,0], dtype=torch.float)] * 100]
	autopilot_targets1 = [2,0]
	domain_args0 = {
		"item_locs":item_locs0,
		"autopilot_utterances": autopilot_utterances0,
		"autopilot_gestures": autopilot_gestures0,
		"autopilot_targets": autopilot_targets0,
		"autopilot":True
	}
	domain_args1 = {
		"item_locs":item_locs1,
		"autopilot_utterances": autopilot_utterances1,
		"autopilot_gestures": autopilot_gestures1,
		"autopilot_targets": autopilot_targets1,
		"autopilot":True
	}
	results0 = run_trials(use_enumeration = True, verbose=False, stop_method = "convergence", domain_args = domain_args0)
	results1 = run_trials(use_enumeration = True, verbose=False, stop_method = "convergence", domain_args = domain_args1)

	# enumeration_beliefs = run_trials(use_enumeration = True, num_trials = num_trials)
	# sans_enumeration_beliefs = run_trials(use_enumeration=False, num_trials=num_trials)

	for t in range(len(autopilot_targets0)):
		print("Domain 0 results for trial {}. Num_steps: {}".format(t,len(results0[t])))
		for step_num, d in enumerate(results0[t]):
			print(d["target_item"])
			print(d["svi_time"])
			print(d["lexicon"])
		print("Domain 1 results for trial {}. Num_steps: {}".format(t,len(results1[t])))

		for step_num, d in enumerate(results1[t]):
			print(d["target_item"])
			print(d["svi_time"])
			print(d["lexicon"])

if __name__ == "__main__":
	# cart2sph_test()
	# sph2cart_test()
	# enumeration_time_test()
	run_trials_independence_test()