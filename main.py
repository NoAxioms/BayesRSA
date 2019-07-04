"""
Generate objects
Pick object
Generate language for picked object
"""
import os, copy, socket, json
from types import SimpleNamespace
from itertools import combinations
import time
from collections import namedtuple
import warnings
# warnings.simplefilter('always')
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
#TODO move step_plate (utterance_plate and gesture_plate) to main_model. Add empty utterances/gestures
#Will need to use cost term. May need to use sampling to avoid combinatorial explosion.
Revelation = namedtuple('Revelation', ['item','word_ids', 'values'])
use_socket = False
if use_socket:
	socket_host = '192.168.0.244'
	socket_port = 8089
	clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	clientsocket.connect(('192.168.0.244', 8089))
#DEBUG PROFILING CODE
time_building_s_0 = 0
time_building_s_2 = 0
#DEBUG_PROFILING_CODE
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

def get_lexicon():
	"""
	Constructs and returns a lexicon tensor
	"""
	num_items = len(Context.words_by_item)
	num_words = len(Context.all_words)
	lexicon = torch.empty(num_items, num_words)
	for i in range(num_items):
		lexicon[i][Context.words_by_item[i]] = torch.tensor(Context.values_by_item[i], dtype=torch.float)
		unkown_words = [w for w in range(num_words) if w not in Context.words_by_item[i]]
		lexicon[i][unkown_words] = pyro.param('vocab_believed_{}'.format(i), torch.ones(len(unkown_words)) * 0.5, constraint=constraints.unit_interval)
	return lexicon
def rsa(s_0, listener_prior=None, depth=1, theta=5., costs = None):
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
		l_1 = torch.log(l_1)
		#Incorporate costs if applicable
		if costs is not None:
			l_1[:] -= costs
		# Update speaker based on listener
		s_2 = softmax(l_1, theta=theta)  # [s][u]  #Should actually softmax logprob
	return s_2

def single_word_model(target_item, items_present, utterances, traj_id, use_rsa=True, theta = 5.):
	global time_building_s_0
	global time_building_s_2
	# print("known words and values:\n{}\n{}".format(known_words_by_item,known_values_by_item))
	num_items = len(items_present)  #Note that this may be less than the number of known items if some are missing from this context
	num_words = len(Context.all_words)
	items_present = torch.tensor(list(items_present))
	#Both item and utterance plates are sequential
	item_plate = pyro.plate('item_plate_{}'.format(traj_id), num_items)
	utterance_plate = pyro.plate('utterance_plate_{}'.format(traj_id), len(utterances))
	#TODO Memoize s_0 calculation
	s_0_start = time.time()
	s_0 = torch.empty((num_items,num_words))
	for i_local_id in item_plate:
		# print("s_0[{}]".format(i_local_id))
		#Get the global id of the item, in case not all items are present
		i_global_id = items_present[i_local_id]
		s_0[i_local_id][Context.words_by_item[i_global_id]] = torch.tensor(Context.values_by_item[i_global_id], dtype=torch.float)
		# print(s_0[i_local_id])
		unkown_words = [w for w in range(num_words) if w not in Context.words_by_item[i_global_id]]
		s_0[i_local_id][unkown_words] = pyro.param('vocab_believed_{}'.format(i_global_id), torch.ones(len(unkown_words)) * 0.5, constraint=constraints.unit_interval)
		# print(s_0[i_local_id])
	s_0_end_time = time.time()
	time_building_s_0 += s_0_end_time - s_0_start
	# target_item_local_id = items_present.index(target_item)
	target_item_local_id = tensor_index(target_item,items_present)
	s_0_local = s_0[items_present]
	s_2 = rsa(s_0=s_0_local, theta=theta) if use_rsa else s_0_local
	time_building_s_2 += time.time() - s_0_end_time
	for u_id in utterance_plate:
		pyro.sample('utterance_{}_{}'.format(traj_id, u_id), dist.Categorical(s_2[target_item_local_id]), obs=torch.tensor(utterances[u_id]))
	return s_0


def single_word_guide(target_item, items_present, utterances, traj_id, use_rsa=True, theta= 5.):
	pass

def bag_of_words_model(target_item, items_present, utterances, traj_id, use_rsa=True, theta = 5., max_words_per_utterance = 3, cost_per_word = 1):
	#DEBUG PROFILING CODE
	global time_building_s_0
	global time_building_s_2
	# print("known words and values:\n{}\n{}".format(known_words_by_item,known_values_by_item))
	num_items = len(items_present)  #Note that this may be less than the number of known items if some are missing from this context
	num_words = len(Context.all_words)
	items_present = torch.tensor(list(items_present))
	#Generate multi-word utterances and their costs (Should we generate these per-item? This would give us fewer, but may not make sense if we don't know lexicon
	max_words_per_utterance = min(max_words_per_utterance,3)
	#TODO check that this gives correct result
	utterances_by_length =[combinations(range(num_words),i) for i in range(max_words_per_utterance + 1)]
	#TODO actually use these utterances. First need to refactor rsa to use cost.
	#Both item and utterance plates are sequential
	item_plate = pyro.plate('item_plate_{}'.format(traj_id), num_items)
	utterance_plate = pyro.plate('utterance_plate_{}'.format(traj_id), len(utterances))
	#TODO Memoize s_0 calculation
	#DEBUG PROFILING CODE
	s_0_start = time.time()
	s_0 = torch.empty((num_items,num_words))
	for i_local_id in item_plate:
		# print("s_0[{}]".format(i_local_id))
		#Get the global id of the item, in case not all items are present
		i_global_id = items_present[i_local_id]
		s_0[i_local_id][Context.words_by_item[i_global_id]] = torch.tensor(Context.values_by_item[i_global_id], dtype=torch.float)
		# print(s_0[i_local_id])
		unkown_words = [w for w in range(num_words) if w not in Context.words_by_item[i_global_id]]
		s_0[i_local_id][unkown_words] = pyro.param('vocab_believed_{}'.format(i_global_id), torch.ones(len(unkown_words)) * 0.5, constraint=constraints.unit_interval)
		# print(s_0[i_local_id])
	#DEBUG PROFILING CODE
	s_0_end_time = time.time()
	time_building_s_0 += s_0_end_time - s_0_start
	# target_item_local_id = items_present.index(target_item)
	target_item_local_id = tensor_index(target_item,items_present)
	s_0_local = s_0[items_present]
	s_2 = rsa(s_0=s_0_local, theta=theta) if use_rsa else s_0_local
	#DEBUG PROFILING CODE
	time_building_s_2 += time.time() - s_0_end_time
	for u_id in utterance_plate:
		pyro.sample('utterance_{}_{}'.format(traj_id, u_id), dist.Categorical(s_2[target_item_local_id]), obs=torch.tensor(utterances[u_id]))
	return s_0

def gesture_model(target_item, gestures, traj_id, arm_length = 0.5, noise = .1):
	"""
	:param target: target_location - human_location
	"""
	#TODO add possibility of None gesture (see gestures.py)
	gesture_plate = pyro.plate('gesture_plate_{}'.format(traj_id), len(gestures))
	target_loc = Context.item_locs[target_item]
	for g_id in gesture_plate:
		head_loc = gestures[g_id][0:3]
		finger_loc = gestures[g_id][3:]  #TODO change to use MR style gestures
		# print("Finger_loc: {}".format(finger_loc))
		# print("Head_loc: {}".format(head_loc))
		cur_gesture = cart2sph(finger_loc - head_loc)
		ideal_vector = cart2sph(target_loc - head_loc)
		dist2target = ideal_vector[2]
		# ideal_vector[2] = arm_length
		covariance_matrix = torch.eye(3)
		covariance_matrix[2] = 0.001
		covariance_matrix *= noise * (dist2target - arm_length)  

		distance = torch.norm(ideal_vector,2)
		cur_gesture = pyro.sample('gesture_{}_{}'.format(traj_id,g_id), dist.MultivariateNormal(loc=ideal_vector, covariance_matrix = torch.eye(3) * noise * distance), obs = cur_gesture)
	# return gesture

def gesture_guide(target_item, gestures, traj_id, arm_length = 0.5, noise = .1):
	pass
def main_model(trajectories, infer = {"enumerate":"parallel"}):
	"""
	:param trajectories: [target_item_id or string for pyro.param belief],[items present], [utterances],[gestures] (in the future, actions)
	"""
	# target_item = pyro.sample('target_item', dist.Categorical(pyro.param('item_probs')))
	# target_loc = Context.item_locs[target_item]
	# gesture = gesture_model(target_loc, head_loc, obs_gesture, arm_length = 0.5, noise = .1)
	for traj_id in pyro.plate('trajectory_plate',len(trajectories)):
		target_item, items_present, utterances, gestures = trajectories[traj_id]
		target_belief_name = None
		#If we don't know the target item, sample it
		if type(target_item) is str:
			target_belief_name = target_item
			target_probs = pyro.param(target_belief_name, torch.ones(len(items_present))/len(items_present), constraint = constraints.simplex)
			target_item = pyro.sample('target_item_{}'.format(traj_id), dist.Categorical(target_probs), infer=infer)
		#Sample language
		single_word_model(target_item, items_present, utterances, traj_id)
		#Sample gesture
		gesture_model(target_item, gestures, traj_id) #gestures = [head position] + [finger position]

def main_guide(trajectories,infer={"enumerate":"parallel"}, **kwargs,):
	for traj_id in pyro.plate('trajectory_plate',len(trajectories)):
		target_item, items_present, utterances, gestures = trajectories[traj_id]
		target_belief_name = None
		#If we don't know the target item, sample it
		if type(target_item) is str:
			target_belief_name = target_item
			target_probs = pyro.param(target_belief_name, torch.ones(len(items_present))/len(items_present), constraint = constraints.simplex)
			target_item = pyro.sample('target_item_{}'.format(traj_id), dist.Categorical(target_probs), infer=infer)
		#Both of the following guide functions are empty, but we call them out of principle.
		single_word_guide(target_item, items_present, utterances, traj_id)
		gesture_guide(target_item,gestures, traj_id)

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
	if type(words) is str:
		words = [words]
		values = [values]
	new_words_ids = []
	new_values = []
	for w_id_in_revelation, w in enumerate(words):
		# print('w: {}'.format(w))
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
def add_words(words):
	new_words = [w for w in words if w not in Context.all_words]
	# num_old_words = len(Context.all_words)
	num_new_words = len(new_words)
	num_items = len(Context.words_by_item)
	for item in range(num_items):
		#Update the vocab_believed
		param_store = pyro.get_param_store()
		vocab_believed = pyro.param('vocab_believed_{}'.format(item), torch.ones(len(Context.all_words)) * 0.5, constraint=constraints.unit_interval)
		num_unknown_old_words = vocab_believed.shape[0]
		vocab_believed_new = torch.ones(num_unknown_old_words + num_new_words) * 0.5
		vocab_believed_new[0:num_unknown_old_words] = vocab_believed
		param_store.__delitem__('vocab_believed_{}'.format(item))
		param_store.setdefault('vocab_believed_{}'.format(item), vocab_believed_new, constraint=constraints.unit_interval)
	Context.all_words.extend(new_words)
def update_with_revelations(svi, revelations, time_limit, svi_args):
	"""
	Based on observation, will update knowledge if applicable, then run SVI on the remaining belief until
	time_limit is reached (TODO end early if it converges)
	"""
	start_time = time.time()
	for r in revelations:
		word_ids_to_remove = update_knowledge(r.item, r.word_ids, r.values)
		update_beliefs(r.item, word_ids_to_remove)
	while time.time() - start_time < time_limit:
		svi.step(**svi_args)
	# svi_args["verbose"] = True
def initialize_knowledge(num_items, item_locs):
	Context.words_by_item = [[] for _ in range(num_items)]
	Context.values_by_item = [[] for _ in range(num_items)]
	Context.item_locs = torch.tensor(item_locs, dtype=torch.float) if type(item_locs) is not torch.Tensor else item_locs  #Should we copy the tensor for safety? Nah


def send_message(target_item, lexicon, action):
	m = {
		'target_item':target_item.tolist(),
		'lexicon':lexicon.tolist(),
		'action':action
	}
	clientsocket.send(bytes(json.dumps(m),'UTF-8'))

#TODO split args into inference/algorithm args and domain args
def run_trials(domain_args, stop_method = 'time', svi_time = 1, convergence_threshold = 0.001, num_trials = 2, use_enumeration = True, verbose = False):
	"""
	Runs trials and returns beliefs at each step of each trial.
	TODO: return observations, actions as well as beliefs.
	:param domain_args: domain arguments (dictionary)
	"""
	global time_building_s_0
	global time_building_s_2
	trials_start_time = time.time()
	da = SimpleNamespace(**domain_args)
	num_items = da.item_locs.shape[0]
	pyro.clear_param_store()
	param_store = pyro.get_param_store()
	#Set Context vars
	all_words = Context.all_words = []
	Context.words_by_item = []
	Context.values_by_item = []
	context = Context(items=tuple([i for i in range(num_items)]))
	context_list = [context]
	#Set up starting knowledge/belief
	initialize_knowledge(num_items, item_locs= da.item_locs)

	trajectories = []  #[target_item_id or string for pyro.param belief],[items present], [utterances],[gestures]
	beliefs_by_trial = []
	adam_params = {"lr": 0.05, "betas": (0.95, 0.999)}
	optimizer = Adam(adam_params)
	if use_enumeration:
		svi = SVI(main_model, main_guide, optimizer, loss=TraceEnum_ELBO(strict_enumeration_warning=False))
		infer_dict = {"enumerate":"parallel"}
	else:
		svi = SVI(main_model, main_guide, optimizer, loss=Trace_ELBO())
		infer_dict = {}
	for trial_num in range(num_trials):
		beliefs_this_trial = []
		if verbose: print("trial_num: {}".format(trial_num))
		#Reset target
		cur_traj = ["target_item_{}_belief".format(trial_num), context.items, [],[]]
		utterances, gestures = cur_traj[2:]
		trajectories.append(cur_traj)
		#Store these words until we know which object they describe
		# words_heard_this_trial = []
		target_item = None
		trial_terminated = False
		trial_step = 0
		while not trial_terminated:
			beliefs_this_step = {}
			if verbose:
				print("trial_step: {}".format(trial_step))
				print("Lexicon:\n{}".format(get_lexicon()))
			if da.autopilot and trial_step < len(da.autopilot_utterances[trial_num]):
				words = da.autopilot_utterances[trial_num][trial_step]
				if verbose: print("You said: {}".format(words))
			else:
				words = input("Speak: ").split(" ")
			add_words(words)
			# all_words.extend([w for w in words if w not in all_words]) #TODO check that lexicon belief accounts for new words
			if da.autopilot and trial_step < len(da.autopilot_gestures[trial_num]):
				gesture = da.autopilot_gestures[trial_num][trial_step]
				if verbose: print("You pointed: {}".format(gesture))
			else:
				gesture = torch.tensor([float(x) for x in input("Point: ").split()], dtype=torch.float)
			#FOr now, assume words has single word. Fix later
			utterances.append(all_words.index(words[0]))
			gestures.append(gesture)
			#Infer
			#TODO use convergence criterion so I can use time tests. Alternatively, do kl tests.
			svi_start_time = time.time()	
			if stop_method == "time":
				while time.time() - svi_start_time < svi_time:
					svi.step(trajectories=trajectories, infer=infer_dict)
			elif stop_method == "convergence":
				#NOTE this is garbage and is only being used for time comparisons for now
				#If item belief and lexicon are stable, we are converged
				#If they are unstable but have stable rolling average?
				target_item_belief = pyro.param(cur_traj[0],torch.ones(num_items)/num_items, constraint = constraints.simplex)
				while target_item_belief.max() < 0.9:
					svi.step(trajectories=trajectories, infer=infer_dict)
					target_item_belief = pyro.param(cur_traj[0],torch.ones(num_items)/num_items, constraint = constraints.simplex)
			else:
				raise Exception("stop_method {} is unimplemented".format(stop_method))
			beliefs_this_step["svi_time"] = time.time() - svi_start_time
			target_item_belief = pyro.param(cur_traj[0])
			lexicon = get_lexicon()
			if verbose:
				print("Lexicon:\n{}".format(lexicon))
				print("new target belief:\n{}".format(target_item_belief))

			action = "pick nose"
			#Log beliefs
			beliefs_this_step["target_item"] = target_item_belief.clone().detach()
			beliefs_this_step["lexicon"] = lexicon
			beliefs_this_trial.append(beliefs_this_step)

			if use_socket: send_message(target_item_belief,lexicon,action)
			#Act
			if target_item_belief.max() > 0.9:
				trial_terminated = True
				if da.autopilot and trial_num < len(da.autopilot_targets):
					target_item = da.autopilot_targets[trial_num]
				else:
					target_item = input("CHEAT: What item: ")
			trial_step += 1
		beliefs_by_trial.append(beliefs_this_trial)
		param_store.__delitem__('target_item_{}_belief'.format(trial_num))
		cur_traj[0] = target_item
		words_heard_this_trial = list(set([u for u in utterances]))  #TODO change to work for set of words utterances
		if verbose: print("words_heard_this_trial: {}".format(words_heard_this_trial))
		revelations = [Revelation(target_item, words_heard_this_trial, [1 for _ in words_heard_this_trial])]	
		update_with_revelations(svi=svi, revelations = revelations, svi_args={'trajectories':trajectories}, time_limit=svi_time)
	trials_total_time = time.time() - trials_start_time
	print("Total trial time: {}\ns_0 time: {}\ns_2 time: {}".format(trials_total_time,time_building_s_0,time_building_s_2))
	return beliefs_by_trial
def act(target_item_belief):
	#TODO make less stupid. Incorporate lexicon, item locations
	max_prob_id = target_item_belief.argmax()
	max_prob = target_item_belief[max_prob_id]
	if max_prob > 0.9:
		return 'pick {}'.format(max_prob_id)
	else:
		return 'point {}'.format(max_prob_id)
if __name__ == "__main__":
	# att_set_test()
	run_trials()
