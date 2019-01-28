import time
import numpy as np
from rsaClass import RSA
from utilities import bayes_rule, uniform
from item_generator import generate_items
import matplotlib.pyplot as plt
#TODO create visualizer for items, dress up game like. Maybe toggle layers in inkscape/blender.
class BayesFilter():
	def __init__(self, item_tuple, transitionMatrix=None, **rsa_kwargs):
		"""
		"""
		self.vocab = tuple(sorted(set((x for i in item_tuple for x in i))))
		self.item_tuple = item_tuple
		self.rsa_kwargs = rsa_kwargs
		self.num_states = len(item_tuple)
		if transitionMatrix is None:
			self.transitionMatrix = np.identity(len(item_tuple))
		else:
			self.transitionMatrix = transitionMatrix #[s][s']
	def simulate(self,s_id,depth=1,b=None, verbose=True):
		if b is None:
			b = uniform(self.num_states)
		for d in range(depth):
			observationMatrix = self.getObservationMatrix(b)
			#Sample observation
			o_id = np.random.choice(observationMatrix.shape[1],p=observationMatrix[s_id])
			if verbose:
				print("o_{}: {}".format(d,self.vocab[o_id]))
			#Update based on transition and observation
			b = self.update(b,o_id)
			if verbose:
				print("b_{}: {}".format(d,b))
		return b
	def update(self,b,o_id):
		if type(o_id) is not int:
			raise TypeError('Need index of observation')
		observationMatrix = self.getObservationMatrix(b)
		#Predict
		b = np.einsum('s,st->t',b,self.transitionMatrix)
		#Update based on observation. Currently computes for all possible observations, need to write a more general bayes rule
		b = bayes_rule(b,observationMatrix)[o_id]  #P(b), P(a | b) [b][a] -> P(b | a) [a][b]
		return b
	def getObservationMatrix(self,belief):
		rsa = RSA(item_tuple = self.item_tuple, **self.rsa_kwargs)
		return rsa.run()
if __name__ == "__main__":
	np.set_printoptions(precision=3,suppress=True)
	# faces_classic = (("face",),("face","moustache"),("face","moustache","glasses"))
	# faces_sym = (("face","moustache"), ("face","moustache","glasses"),("face", "glasses"))
	items = generate_items()
	print(items)
	rsa_kwargs = {
		"theta":10.0,
		"default_depth":10
	}
	bf = BayesFilter(item_tuple = items,**rsa_kwargs)
	b = bf.simulate(0,depth=10)
	print(b)