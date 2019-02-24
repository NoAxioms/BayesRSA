import json, os, sys
import numpy as np
from rsaClass import RSA
from item_generator import generate_items
from BayesClass import BayesFilter
np.set_printoptions(precision=2,floatmode='fixed')
def word_clearing_test():
	with open("./standard_faces.json", 'r') as f:
		standard_faces = json.load(f)
		faces = tuple(x.split(" ") for x in standard_faces["faces_sequence_alph"])
	rsa = RSA(items=faces, theta=5)
	for i in range(10):
		print("Depth {}".format(i))
		for w_id, w in enumerate(rsa.vocab):
			# print("{}: {}".format(w,rsa.speaker_probs[:,w_id]))
			print("{}: {}".format(w,rsa.listener_probs[w_id]))
		rsa.run(depth=1)

def find_good_parameters():
	# min_depth = 1
	# max_depth = 4
	# num_depth = max_depth - min_depth + 1
	# min_theta = 1
	# max_theta = 10
	# num_theta = max_theta - min_theta + 1

	update_priors = [0]
	depths = list(range(1,30))
	thetas = list(range(1,30))	


	beliefs = np.empty(shape=(len(update_priors),len(thetas),len(depths)))
	item_id = 3
	faces = generate_items(3)
	observations = faces[item_id]
	print("Observations: {}".format(observations))
	for u_id, u in enumerate(update_priors):
		for t_id, t in enumerate(thetas):
			for d_id, d in enumerate(depths):
				rsa_config = {"theta":t, "default_depth":d}
				bf = BayesFilter(items=faces, rsa_config = rsa_config)
				final_belief = bf.simulate(observation_sequence = observations, update_rsa_prior = u)[0][-1]
				beliefs[u_id,t_id,d_id] = final_belief[item_id]
	best_index = np.unravel_index(np.argmax(beliefs),dims=beliefs.shape)
	best_belief = beliefs[best_index]
	print("Update prior: {}\ntheta: {}\ndepth: {}\nbelief: {}".format(update_priors[best_index[0]],thetas[best_index[1]], depths[best_index[2]], best_belief))



if __name__ == "__main__":
	# word_clearing_test()
	find_good_parameters()