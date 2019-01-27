import time
import itertools
import numpy as np
from utilities import softmax, bayes_rule, uniform, arr2tex

def proportional_vector_prob(utterance_vectors, alpha = 0.0):
	"""
	:param alpha: A noise parameter between 0 and 1. alpha=0 is noiseless, alpha=1 gives uniform distributions.
	Assumes no vector is empty.
	:return: P(s | u)=\frac{u(s) + \alpha}{|u| + \alpha|S|} matrix [u][s]
	"""
	#Tested and working
	num_utterances = utterance_vectors.shape[0]
	num_states = utterance_vectors.shape[1]
	numerators = utterance_vectors * (1.0 - alpha) + alpha
	noiseless_denominators = np.sum(utterance_vectors,axis=1) #shape = u
	noisy_denominators = noiseless_denominators* (1-alpha) + alpha * num_states
	noisy_denominators = noisy_denominators.repeat(num_states).reshape((num_utterances,num_states))
	# print("noise_denominators: \n{}".format(noisy_denominators))
	probs = np.true_divide(numerators,noisy_denominators)
	return probs

def binary_listener_probs(items):
	"""
	return: probs [u][s], vocab
	"""
	vocab = tuple(sorted(set((x for i in items for x in i))))
	probs = np.zeros(shape=(len(vocab), len(items)))
	for v_id, v in enumerate(vocab):
		for i_id, i in enumerate(items):
			if v in i:
				probs[v_id,i_id] = 1.0
			else:
				probs[v_id,i_id] = 0.0
	#normalize
	denominators = np.sum(probs,axis=1)
	probs = np.true_divide(probs.swapaxes(0,1),denominators).swapaxes(0,1)
	return probs, vocab
class RSA():
	def __init__(self, listener_probs_lit=None, listener_prior = None, theta=1.0, item_tuple = None):
		if item_tuple is not None:
			listener_probs_lit, vocab = binary_listener_probs(item_tuple)
			self.vocab = vocab
		self.listener_probs_lit = listener_probs_lit
		self.listener_probs = self.listener_probs_lit
		self.theta = theta
		self.speaker_probs = softmax(self.listener_probs,axis=0,theta=self.theta).swapaxes(0,1) #[s][u]
		self.num_states = listener_probs_lit.shape[1]
		self.num_utterances = listener_probs_lit.shape[0]
		self.cur_depth = 0
		self.running_time = 0
		if listener_prior is None:
			self.listener_prior = np.array([1.0 / self.num_states for i in range(self.num_states)])
		else:
			self.listener_prior = listener_prior
	def run(self,depth):
		start_time = time.time()
		for d in range(depth):
			#Update listener based on speaker: P_l(s | w, a) prop P_s(w | s, a)P(s)
			self.listener_probs = bayes_rule(self.listener_prior,self.speaker_probs) #[u][s]
			#Update speaker based on listener
			self.speaker_probs = softmax(self.listener_probs,axis=0,theta=self.theta).swapaxes(0,1) #[s][u]
			self.cur_depth += 1
		self.running_time += time.time() - start_time
	def reset(self):
		self.listener_probs = self.listener_probs_lit
		self.speaker_probs = softmax(self.listener_probs,axis=0,theta=self.theta).swapaxes(0,1) #[s][u]
		self.cur_depth = 0
		self.running_time = 0
def rsa_test(item_tuple, depth=1, priors = None, theta=5.0):
	print("theta: {}".format(theta))
	if priors is None:
		priors = (uniform(len(item_tuple)),)
	rsa_list = []
	for p in priors:
		rsa = RSA(item_tuple=item_tuple,theta=theta,listener_prior = p)
		print("Vocab: {}".format(rsa.vocab))
		print("Items: {}".format(item_tuple))
		print("Prior: {}".format(p))
		# print("\nListener_0 [u][s]:\n{}".format(rsa.listener_probs))
		print("\nSpeaker_0 [s][u]:\n{}".format(rsa.speaker_probs))
		rsa.run(depth=depth)
		print("\nListener_{} [u][s]:\n{}".format(depth, rsa.listener_probs))
		print("\nSpeaker_{} [s][u]:\n{}".format(depth, rsa.speaker_probs))
		rsa_list.append(rsa)
	return rsa_list
def test_permutation_invariance():
	faces_classic = (("face",),("face","moustache"),("face","moustache","glasses"))
	permutations = list(itertools.permutations([i for i in range(len(faces_classic))]))
	rsa_list = []
	for perm in permutations:
		permuted_items = tuple((faces_classic[perm[i]] for i in range(len(perm))))
		print(permuted_items)
		rsa_list.append(rsa_test(permuted_items,depth=10)[0])
	speaker_probs_standard = rsa_list[0].speaker_probs
	for perm_id, perm in enumerate(permutations):
		#Compare to first permutation
		de_permuted_probs = np.empty(shape=(speaker_probs_standard.shape))
		permuted_probs = rsa_list[perm_id].speaker_probs
		print("\npermutation: {}".format(perm))
		print("standard_probs: \n{}".format(speaker_probs_standard))
		print("permuted_probs: \n{}".format(permuted_probs))
		for s_id, s in enumerate(perm):
			de_permuted_probs[s,:] = permuted_probs[s_id,:]

		assert np.array_equal(de_permuted_probs,speaker_probs_standard), "{} {}\n{}\n\n{}".format(perm_id,perm,speaker_probs_standard,de_permuted_probs)

faces_classic = (("face",),("face","moustache"),("face","moustache","glasses"))
faces_sym = (("face","moustache"), ("face","moustache","glasses"),("face", "glasses"))
faces_sym2 = (("face", "glasses"),("face","moustache","glasses"),("face","moustache"))
item_tuples_tuple = (faces_sym,faces_sym2)
np.set_printoptions(precision=3,suppress=True)
priors = (uniform(3),np.array((0.01,0.495,0.495)))
# priors = (np.array((0.01,0.495,0.495)),)
rsa_test(faces_sym,depth=1000,theta=10.0,priors=priors)
# rsa_test(faces_sym2,depth=10,theta=10.0,priors=priors)
# test_permutation_invariance()