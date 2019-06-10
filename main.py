"""
Generate objects
Pick object
Generate language for picked object
"""
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
#TODO change RSA to use whole utterance. 
#Will need to use cost term. May need to use sampling to avoid combinatorial explosion.
Revelation = namedtuple('Revelation', ['item','word_ids', 'values'])

class Context():
	"""
	A context is a set of items present, and a count of words attributed to each item
	To incorporate gesture, we will need to track item location. 
	If we track (utterance, gesture) pairs, then a count will no longer be appropriate since
	gestures are continuous valued and thus usually unique
	"""
	#Set all_words to the correct list.
	all_words = []
	words_by_item = []
	values_by_item = []
	def __init__(self, items):
		self.items = copy.copy(items)
		self.word_counts = torch.zeros(len(self.items), len(Context.all_words))
	def hear(self,item, word):
		word_id = Context.all_words.index(word)
		# print("word, id: {}, {}".format(word, word_id))
		try:
			self.word_counts[item][word_id] += 1
		except IndexError:
			#TODO find pretty, pytorchy way to do this assignment
			#Create new array of correct size, copy data from old word_counts
			new_word_counts = torch.zeros(num_items, len(all_words))
			new_word_counts[:][0:self.word_counts.shape[1]] = self.word_counts
			self.word_counts = new_word_counts


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
		s_2 = softmax(l_1, theta=theta)  # [s][u]  #Should actually softmax logprob
	return s_2

def language_model(contexts, num_items, use_rsa=True, theta = 5.):
	# print("known words and values:\n{}\n{}".format(known_words_by_item,known_values_by_item))
	num_words = len(Context.all_words)
	item_plate = pyro.plate('item_plate', num_items)
	s_0 = torch.empty((num_items,num_words))
	for i in item_plate:
		# print("s_0[{}]".format(i))
		s_0[i][Context.words_by_item[i]] = torch.tensor(Context.values_by_item[i], dtype=torch.float)
		# print(s_0[i])
		unkown_words = [w for w in range(num_words) if w not in Context.words_by_item[i]]
		s_0[i][unkown_words] = pyro.param('vocab_believed_{}'.format(i), torch.ones(len(unkown_words)) * 0.5, constraint=constraints.unit_interval)
		# print(s_0[i])
	for c_id, c in enumerate(contexts):
		s_0_local = s_0[list(c.items)]
		assert s_0_local.shape == (len(c.items), num_words), "{};   {}".format(s_0_local.shape, (len(c.items), num_words))
		s_2 = rsa(s_0=s_0_local, theta=theta) if use_rsa else s_0_local
		for i in item_plate:
			pyro.sample('word_count_{}_{}'.format(c_id,i), dist.Multinomial(
				total_count=c.word_counts[i].sum().item(), probs=s_2[i]),obs=c.word_counts[i])
	return s_0

def language_guide(contexts, num_items, use_rsa=True, theta= 5.):
	# print("known words and values:\n{}\n{}".format(known_words_by_item,known_values_by_item))
	num_words = len(Context.all_words)
	item_plate = pyro.plate('item_plate', num_items)
	s_0 = torch.empty((num_items,num_words))
	for i in item_plate:
		# print("s_0[{}]".format(i))
		s_0[i][Context.words_by_item[i]] = torch.tensor(Context.values_by_item[i], dtype=torch.float)
		# print(s_0[i])
		unkown_words = [w for w in range(num_words) if w not in Context.words_by_item[i]]
		s_0[i][unkown_words] = pyro.param('vocab_believed_{}'.format(i), torch.ones(len(unkown_words)) * 0.5,constraint=constraints.unit_interval)
		# print(s_0[i])
	return s_0

def gesture_model(target_loc, head_loc, obs_gestures, arm_length = 0.5, noise = .1):
	"""
	:param target: target_location - human_location
	"""
	ideal_vector = cart2sph(target_loc - head_loc)
	dist2target = ideal_vector[2]
	# ideal_vector[2] = arm_length
	covariance_matrix = torch.eye(3)
	covariance_matrix[2] = 0.001
	covariance_matrix *= noise * (dist2target - arm_length)  

	distance = torch.norm(ideal_vector,2)
	gesture = pyro.sample('gesture', dist.MultivariateNormal(loc=ideal_vector, covariance_matrix = torch.eye(3) * noise * distance), obs = obs_gesture)
	return gesture

def trial_model(item_locs, head_loc, obs_gestures, obs_words, arm_length = 0.5, noise = .1):
	target_item = pyro.sample('target_item', dist.Categorical(pyro.param('item_probs')))
	target_loc = item_locs[target_item]
	gesture = gesture_model(target_loc, head_loc, obs_gesture, arm_length = 0.5, noise = .1)

def trial_guide(item_locs, head_loc, obs_gestures, obs_words, arm_length = 0.5, noise = .1):
	target_item = pyro.sample('target_item', dist.Categorical(pyro.param('item_probs')))

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

def update_knowledge(item, words, values):
	new_words_ids = []
	new_values = []
	for w_id_in_revelation, w in enumerate(words):
		print('w: {}'.format(w))
		if w not in Context.words_by_item[item]:
			new_words_ids.append(w)
			new_values.append(values[w_id_in_revelation])
		#If we already know the value of the word, we were wrong. Update it.
		else:
			w_index_in_known = Context.words_by_item[item].index(w)
			current_val = Context.values_by_item[item][w_index_in_known]
			new_val = values[w_id_in_revelation]
			if current_val != new_val:
				print("Certain knowledge was wrong: {}[{}]: {} -> {}".format(item,w,current_val,new_val))
				Context.values_by_item[item][w_index_in_known] = new_val
	Context.words_by_item[item].extend(new_words_ids)
	Context.values_by_item[item].extend(new_values)
	return new_words_ids  #return this to use in updating belief params

def update_beliefs(item, word_ids_to_remove):
	param_store = pyro.get_param_store()
	vocab_believed = pyro.param('vocab_believed_{}'.format(item), torch.ones(len(Context.all_words)) * 0.5, constraint=constraints.unit_interval)
	new_words_local = [global_2_local_id(w, Context.words_by_item[item]) for w in word_ids_to_remove]
	vocab_believed_new = copy_array_sans_indices(vocab_believed,new_words_local)
	# param_store.replace_param(param_name='vocab_believed_{}'.format(item),new_param=vocab_believed_new,old_param=vocab_believed)  #Add positive constraint
	param_store.__delitem__('vocab_believed_{}'.format(item))
	param_store.setdefault('vocab_believed_{}'.format(item), vocab_believed_new, constraint=constraints.unit_interval)
	# param_store.__setitem__('vocab_believed_{}'.format(item), vocab_believed_new)
	return pyro.param('vocab_believed_{}'.format(item))

def update_with_revelations(svi, revelations, time_limit, svi_args):
	"""
	Based on observation, will update knowledge if applicable, then run SVI on the remaining belief until
	time_limit is reached (TODO end early if it converges)
	"""
	start_time = time.time()
	for r in revelations:
		word_ids_to_remove = update_knowledge(r.item, r.words, r.values)
		update_beliefs(r.item, word_ids_to_remove)
	while time.time() - start_time < time_limit:
		svi.step(**svi_args)
	# svi_args["verbose"] = True
def initialize_knowledge(num_items):
	Context.words_by_item = [[] for _ in range(num_items)]
	Context.values_by_item = [[] for _ in range(num_items)]
def att_set_test():
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

def run_trials():
	num_items = 3
	initialize_knowledge(num_items)
	# num_words = 0
	all_words = []
	Context.all_words = all_words
	context = Context(items=(0,1,2))
	context_list = [context]
	pyro.clear_param_store()
	adam_params = {"lr": 0.05, "betas": (0.95, 0.999)}
	optimizer = Adam(adam_params)
	svi = SVI(language_model, language_guide, optimizer, loss=Trace_ELBO())
	num_trials = 5
	for trial_num in range(num_trials):
		#Store these words until we know which object they describe
		words_heard_this_trial = []
		target_item = None
		trial_terminated = False
		while not trial_terminated:
			words = raw_input("Speak: ")
			gesture = raw_input("Point: ")
			#Update belief over target_item

			#Act

			
		for w in words_heard_this_trial:
			context.hear(target_item, w)


	revelations = []
	revelations.append(Revelation(0,[0],[1]))
	revelations.append(Revelation(1,[1],[1]))
	svi_args = {
		'contexts':context_list,
		'num_items':num_items,
		'use_rsa':True
	}
	time_limit = 3
	context.hear(0,0)
	update_with_revelations(svi=svi, revelations=[revelations[0]],svi_args=svi_args, time_limit=time_limit)
	context.hear(1,1)
	update_with_revelations(svi=svi, revelations=[revelations[1]],svi_args=svi_args, time_limit=time_limit)
	print(language_model(context_list, num_items))

if __name__ == "__main__":
	att_set_test()
