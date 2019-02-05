import time
import numpy as np
from utilities import softmax, bayes_rule, uniform, arr2tex, get_belief_grid
from display import plotBeliefSimplex

class RSA():
	def __init__(self, listener_probs_lit=None, listener_prior = None, theta=1.0, items = None, vocab=None, default_depth=1):
		if items is not None:
			listener_probs_lit, vocab = binary_listener_probs(items)
			vocab = vocab
			items = items
		self.vocab = vocab
		self.items = items
		self.listener_probs_lit = listener_probs_lit
		self.listener_probs = self.listener_probs_lit
		self.theta = theta
		self.speaker_probs = softmax(self.listener_probs,axis=0,theta=self.theta).swapaxes(0,1) #[s][u]
		self.num_states = listener_probs_lit.shape[1]
		self.num_utterances = listener_probs_lit.shape[0]
		self.default_depth = default_depth
		self.cur_depth = 0
		self.running_time = 0
		if listener_prior is None:
			self.listener_prior = np.array([1.0 / self.num_states for i in range(self.num_states)])
		else:
			self.listener_prior = listener_prior
	def run(self,depth=None):
		start_time = time.time()
		if depth is None:
			depth=self.default_depth
		for d in range(depth):
			#Update listener based on speaker: P_l(s | w, a) prop P_s(w | s, a)P(s)
			self.listener_probs = bayes_rule(self.listener_prior,self.speaker_probs) #[u][s]
			#Update speaker based on listener
			self.speaker_probs = softmax(self.listener_probs,axis=0,theta=self.theta).swapaxes(0,1) #[s][u]
			self.cur_depth += 1
		self.running_time += time.time() - start_time
		return self.speaker_probs
	def reset(self):
		self.listener_probs = self.listener_probs_lit
		self.speaker_probs = softmax(self.listener_probs,axis=0,theta=self.theta).swapaxes(0,1) #[s][u]
		self.cur_depth = 0
		self.running_time = 0
	def __str__(self):
		s = "Vocab: {}".format(self.vocab)
		s += "\nItems: {}".format(self.items)
		s += "\nPrior: {}".format(self.listener_prior)
		s += "\nSpeaker_{} [s][u]:\n{}".format(self.cur_depth,self.speaker_probs)
		s += "\nListener_{} [u][s]:\n{}".format(self.cur_depth,self.listener_probs)
		return s
class RSAExplorer():
	"""
	Run RSA with different parameters and plot results. Currently only works with dimension 3
	"""
	def __init__(self, items, theta = 1.0, rsa_depth = 5):
		self.listener_probs_lit, self.vocab = binary_listener_probs(items)
		self.items = items
		self.theta = theta
		belief_resolution = len(items) * 10
		# self.belief_points = freudenthal(len(items), upper_bound=freudenthal_bound)
		self.belief_points = get_belief_grid(len(items), resolution=belief_resolution)
		print(self.belief_points)
		print("number beliefs: {}".format(len(self.belief_points)))
		self.rsa_args = {"listener_probs_lit": self.listener_probs_lit, "theta":self.theta, "default_depth":rsa_depth}
		self.gatherSpeakerProbs()
		self.display()
	def gatherSpeakerProbs(self):
		self.speaker_probs = np.empty(shape=(len(self.belief_points), len(self.items),len(self.vocab))) #b_i, s, u
		for b_id in range(len(self.belief_points)):
			rsa = RSA(listener_prior=self.belief_points[b_id], **self.rsa_args)
			self.speaker_probs[b_id,:,:] = rsa.run()
		return self.speaker_probs
	def display(self, u_id = 0, condition_states = (0,1)):
		speaker_ratios = np.true_divide(self.speaker_probs[:,condition_states[0],u_id],self.speaker_probs[:,condition_states[1],u_id])
		title = "P({} | {}) / P({} | {})".format(self.vocab[u_id],self.items[condition_states[0]], self.vocab[u_id],self.items[condition_states[1]])
		axis_labels = (str(self.items[0]), str(self.items[1]))
		plotBeliefSimplex(self.belief_points,speaker_ratios, title=title, axis_labels=axis_labels)

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