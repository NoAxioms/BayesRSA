import time
import numpy as np

class BayesFilter():
	def __init__(self, observation_function, transition_function):
		"""
		"""
		self.observation_function = observation_function
		self.transition_function = transition_function
	def run(self,depth):
		start_time = time.time()
		for d in range(depth):
			pass
		self.running_time += time.time() - start_time