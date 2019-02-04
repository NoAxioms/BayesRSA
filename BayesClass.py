import time, os, datetime, json
import numpy as np
from rsaClass import RSA
from utilities import bayes_rule, uniform
from item_generator import generate_items
import matplotlib.pyplot as plt
from display import createBeliefFigure
# np.random.seed(seed=0)
#TODO 
class BayesFilter():
	def __init__(self, items, transitionMatrix=None, rsa_config=None, name=None):
		"""
		"""
		self.name = name if name is not None else str(datetime.datetime.now()).replace(":","_").replace(" ","_")[5:13]
		self.rsa_config = rsa_config if rsa_config is not None else {}
		self.record_path = os.path.dirname(os.path.realpath(__file__)) + "/trajectory_logs/" + self.name + "/"
		self.vocab = tuple(sorted(set((x for i in items for x in i))))
		self.items = items
		self.num_states = len(items)
		self.transitionMatrix = transitionMatrix if transitionMatrix is not None else np.identity(len(items)) #[s][s']
		self.num_simulations = 0

	def simulate(self,s_id,depth=1,b=None, save_trajectories = True, save_images = False, update_rsa_prior = True):
		"""
		TODO: Allow beliefs to be viewed while simulation is still running. 
		"""
		if b is None:
			b = uniform(self.num_states)
		b0 = b
		belief_list = [b]
		o_list = ["START"]
		if (save_images or save_trajectories):
			self.num_simulations += 1
			if not os.path.isdir(self.record_path):	os.mkdir(self.record_path)
		if save_images:
			createBeliefFigure(b,self.items,"START",save_location=self.record_path + "/{}_0.png".format(self.num_simulations - 1), des_id=s_id)
		for d in range(depth):
			observationMatrix = self.getObservationMatrix(b) if update_rsa_prior else self.getObservationMatrix(b0)
			#Sample observation
			o_id = np.random.choice(observationMatrix.shape[1],p=observationMatrix[s_id])
			#Update based on transition and observation
			b = self.update(b,o_id, observationMatrix)
			o_list.append(self.vocab[o_id])
			belief_list.append(b)
			if save_images:
				createBeliefFigure(b,self.items,self.vocab[o_id],save_location=self.record_path + "/{}_{}.png".format(self.num_simulations - 1, d + 1),des_id=s_id)
			# displayDistribution(b,self.items)
		if save_trajectories:
			json_name = "simulation{}.json".format(self.num_simulations - 1)

			data = {
				"items":[" ".join(x) for x in self.items],
				"rsa_config": self.rsa_config,
				"steps":[{"observation":o_list[i], "belief":list(belief_list[i])} for i in range(len(belief_list))]
				}
			with open(self.record_path + json_name,'w') as f:
				json.dump(data,f, indent=4)
		return belief_list, o_list
	def update(self,b,o_id, obs_mat = None):
		if type(o_id) is not int:
			raise TypeError('Need index of observation')
		observationMatrix = self.getObservationMatrix(b) if obs_mat is None else obs_mat
		#Predict
		b = np.einsum('s,st->t',b,self.transitionMatrix)
		#Update based on observation. Currently computes for all possible observations, need to write a more general bayes rule
		b = bayes_rule(b,observationMatrix)[o_id]  #P(b), P(a | b) [b][a] -> P(b | a) [a][b]
		return b
	def getObservationMatrix(self,belief):
		rsa = RSA(items = self.items, **self.rsa_config)
		return rsa.run()
	def item_to_string(self, item):
		"""TODO make face class and put this function, along with the face image functions, there"""
		return " ".join(item)

if __name__ == "__main__":
	np.set_printoptions(precision=3,suppress=True)
	# faces_classic = (("face",),("face","moustache"),("face","moustache","glasses"))
	# faces_sym = (("face","moustache"), ("face","moustache","glasses"),("face", "glasses"))
	items = generate_items()
	print(items)
	rsa_config = {
		"theta":10.0,
		"default_depth":10
	}
	bf = BayesFilter(items = items,rsa_config = rsa_config)
	b = bf.simulate(0,depth=10)
	print(b)