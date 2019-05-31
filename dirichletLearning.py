"""
Generate objects
Pick object
Generate language for picked object
"""
import os
import time
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


def normalize(x):
	"""
	Normalizes 2d tensor so that x.sum(1)[i] = 1.0
	TODO make dim argument
	TODO make work with broadcasting for enumeration stuff
	"""
	assert len(x.shape) == 2, x.shape
	d = x.sum(dim=1).repeat(x.shape[1]).reshape(
		x.shape[1], x.shape[0]).transpose(0, 1)
	assert d.shape == x.shape, "x.shape: {};    d.shape: {}".format(
		x.shape, d.shape)
	return x / d


def softmax(x, theta=1.0):
	# 2d only
	exponentiated = torch.exp(theta * x)
	return normalize(exponentiated)
	# print("exp:\n{}".format(exponentiated))
	# denominators = exponentiated.sum(dim=0)
	# print("den:\n{}".format(denominators))
	# return (exponentiated.transpose(0,1) / denominators).transpose(0,1)


def bayes_rule(b, a_cond_b, a=None, debug=False):
	"""
	:param a: Prob(a) matrix
	:param b: Prob(b) matrix
	:param a_cond_b: prob(a | b) matrix with [b][a] indexing
	:return: prob(b | a) matrix with [b][a] indexing
	"""
	# Calculate a from b and a_cond_b if a=None
	# normalize
	a_cond_b = normalize(a_cond_b)
	b = b / b.sum()
	if a is None:
		a = torch.einsum('b,ba->a', b, a_cond_b)
		if debug:
			print("a:\n", a)
	# b_stretched = b.repeat(a.shape[0]).reshape((b.shape[0],a.shape[0])).transpose(0,1)  #[b][a]
	b_stretched = b.reshape((b.shape[0], 1)).repeat(1, a.shape[0])
	if debug:
		print("a_cond_b:\n", a_cond_b)
		print("b_stretched:\n", b_stretched)
	a_join_b = (a_cond_b * b_stretched)  # [b][a]
	if debug:
		print("a_join_b:\n", a_join_b)
	a_stretched = a.reshape(1, a.shape[0]).repeat(b.shape[0], 1)  # [b][a]
	if debug:
		print("a_stretched:\n", a_stretched)
	b_cond_a = (a_join_b / a_stretched)  # [b][a]
	# If some values of a have prob 0, we get nans. Replace nans with 0.
	nan_id = torch.isnan(b_cond_a)
	b_cond_a[nan_id] = 0.0
	if debug:
		print("b_cond_a:\n", b_cond_a)
		print("nan_id:\n", nan_id)

	return b_cond_a


def rsa(s_0, listener_prior=None, depth=1, theta=5.):
	"""
	:param s_0: [s][u] (does not need to be normalized)
	returns: s_1 [s][u]
	"""
	s_0 = normalize(s_0)
	num_items = s_0.shape[0]
	if listener_prior is None:
		listener_prior = torch.ones(num_items) / num_items

	for d in range(depth):
		# Update listener based on speaker: P_l(s | w, a) prop P_s(w | s, a)P(s)
		l_1 = bayes_rule(listener_prior, s_0)  # [s][u]
		# Update speaker based on listener
		s_2 = softmax(l_1, theta=theta)  # [s][u]
	return s_2


def model(observations=None, use_rsa=True, verbose=True, num_items = 0, vocab_size=0, theta=5., speaker_prior='dirichlet'):
	"""
	:param observations: list of (target, context, word histogram) triples. s_0[context] returns s_0 restricted to present items
	"""
	item_plate = pyro.plate('item_plate', num_items)
	#Initialize s_0
	if speaker_prior == "dirichlet":
		speaker_concentrations = pyro.param("speaker_concentrations", 0.5 * torch.ones(
				num_items, vocab_size), constraint=constraints.positive)
		s_0 = torch.empty(num_items, vocab_size) 
		
		for item_num in item_plate:
			s_0[item_num] = pyro.sample('s_0_{}'.format(
				item_num), dist.Dirichlet(speaker_concentrations[item_num]))
	elif speaker_prior == "attribute_set":
		word_plate = pyro.plate('word_plate', vocab_size)
		#[item][word] == 1 if word is known to apply to item, else 0
		known_words = pyro.param("known_words", torch.zeros(num_items, vocab_size))  #Get rid of gradient here
		#[item][word] == probability word is known to apply to item
		vocab_beliefs = pyro.param('vocab_beliefs', torch.ones(num_items, vocab_size) * 0.5)
		item_vocabs = known_words.clone().detach().requires_grad(True)
		for item_num in item_plate:
			for word_num in word_plate:
				if known_words[item_num][word_num] == 0:
					item_vocabs[item_num][word_num] = pyro.sample("item_vocab_{}_{}".format(item_num,word_num), dist.Bernoulli(vocab_beliefs[item_num][word_num]))
		

	if verbose:
		print("model s_0:\n{}".format(s_0))
	#Iterate through observations
	for obs_id, obs in enumerate(observations):
		target_item, context, word_counts = obs
		#Get index of target_item in the context
		target_item_local = context.index(target_item)
		s_0_local = s_0[context]
		s_2 = rsa(s_0=s_0_local) if use_rsa else s_0_local
		pyro.sample('word_count_{}'.format(obs_id), dist.Multinomial(
			total_count=word_counts.sum().item(), probs=s_2[target_item_local]),obs=word_counts)
	return s_0
	# for target_item in item_plate:
	# 	# print("!!!!!!!!!!!!!!\n{}\n!!!!!!!!!!!!!!!!".format(type(utterances_heard[target_item].sum())))
	# 	# print(isinstance(utterances_heard[target_item].sum(),Number))
	# 	pyro.sample('word_count_{}'.format(target_item), dist.Multinomial(
	# 		total_count=utterances_heard[target_item].sum().item(), probs=s_2[target_item]),obs=utterances_heard[target_item])



def guide(observations=None, use_rsa=True, verbose=True, num_items = 0, vocab_size=0, theta=5.):
	"""
	:param observations: list of (target, context, word histogram) triples. s_0[context] returns s_0 restricted to present items
	"""
	#Initialize s_0
	speaker_concentrations = pyro.param("speaker_concentrations", 0.5 * torch.ones(
			num_items, vocab_size), constraint=constraints.positive)
	s_0 = torch.empty(num_items, vocab_size) 
	item_plate = pyro.plate('item_plate', num_items)
	for item_num in item_plate:
		s_0[item_num] = pyro.sample('s_0_{}'.format(
			item_num), dist.Dirichlet(speaker_concentrations[item_num]))
	if verbose:
		print("guide s_0:\n{}".format(s_0))

	return s_0
def bayes_rule_test():  # pass
	# q = torch.tensor([[1.,1.],[1.,1.]])
	# w = torch.tensor([[2.,3.], [2.,2.]])
	# print(q/w)
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


def l1_distance(a, b):
	return torch.abs(a - b).sum()


def generate_observations(s_0_true=None, num_utterances_per_item=1000, theta=5., context_list = None, skipped_items = ()):
	if s_0_true is None:
		s_0_true = normalize(torch.tensor(
			[[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.float))
	num_items = s_0_true.shape[0]
	if context_list is None:
		context_list = [list(range(num_items))]
	observations = []
	for c in context_list:
		s_0_local = s_0_true[c]
		s_2_true = rsa(s_0_local, theta=theta)
		for target_item_local, target_item in enumerate(c):
			if target_item not in skipped_items:
				word_count = torch.zeros(len(c))
				word_dist = dist.Categorical(s_2_true[target_item_local])
				for utt in range(num_utterances_per_item):
					word = word_dist.sample()
					word_count[word] += 1
				observations.append((target_item,c,word_count))
	return observations

	# 	utterances_by_item = torch.empty(size=(num_items, num_utterances_per_item))
	# 	for target_item in range(num_items):
	# 		for utt in range(num_utterances_per_item):
	# 			utterances_by_item[target_item][utt] = dist.Categorical(
	# 				s_2_true[target_item]).sample()
	# return utterances_by_item.long()


def main():
	use_rsa = False
	theta=10.
	# generate test data
	# utterances_by_item = torch.tensor([[0] * n, [1] * n, [2] * n])
	num_items = 3
	vocab_size=3
	s_0_true = normalize(torch.tensor(
		[[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.float))
	# utterances_by_item = generate_data(s_0_true=s_0_true)

	# max_utterance_id = utterances_by_item.max()
	# print(max_utterance_id)
	# utterance_counts = torch.empty(size=(3, 2 + 1))
	# for item in range(num_items):
	# 	utterance_counts[item] = utterances_by_item[item].bincount()
	# utterance_probs_empirical = normalize(utterance_counts)
	# # TODO make sure no element of utterance_probs_empirical is 0, otherwise we cannot use it as concentration parameter.
	# print("utterance_probs_empirical:\n{}".format(utterance_probs_empirical))

	context_list = [[0,1,2]]
	observations = generate_observations(s_0_true=s_0_true, context_list=context_list,theta=theta, skipped_items = (1,), num_utterances_per_item=1)


	pyro.clear_param_store()
	# Initialize speaker concentrations based on empirical distribution of utterances. (Need to rewrite to deal with the multi-context setting)
	# pyro.param("speaker_concentrations", utterance_probs_empirical,
	# 		   constraint=constraints.positive)
	adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
	optimizer = Adam(adam_params)
	svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
	n_steps = 1000
	# do gradient steps
	start_time = time.time()
	for step in range(n_steps):
		verbose = True and ((not step % 100) or (step == n_steps - 1))
		if step == n_steps - 1:
			print("~~~~~~~~~~~~~")
		if verbose:
			print('step: {}'.format(step))
		# It is not always using the data as observation. Why?
		loss = svi.step(observations = observations,
				 verbose=verbose, use_rsa=use_rsa, num_items=num_items, vocab_size=vocab_size, theta=theta)
		if verbose:
			print("loss: {}".format(loss))
	end_time = time.time()
	# s_2 = guide(use_rsa=use_rsa, utterances_heard=utterances_by_item)
	print("total time: {}".format(end_time - start_time))

	# print("final s_2:\n{}".format(s_2))
if __name__ == "__main__":
	main()
	# elbo = TraceEnum_ELBO(max_plate_nesting=3)
	# elbo.loss(model, config_enumerate(model, "sequential")); #sequenial and parallel break in different ways.
	# elbo = Trace_ELBO()
	# elbo.loss(model, model);
	# rsa_test()