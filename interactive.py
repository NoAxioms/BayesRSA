import json, os, sys
import numpy as np
from item_generator import generate_items
from BayesClass import BayesFilter
np.set_printoptions(precision=2,floatmode='fixed')

if __name__ == "__main__":
	# with open("./standard_faces.json", 'r') as f:
	# 	standard_faces = json.load(f)
	# 	faces = tuple(x.split(" ") for x in standard_faces["faces_classic"])
	faces = generate_items(3)
	rsa_config = {"theta":25, "default_depth":8}

	bf = BayesFilter(items=faces, rsa_config=rsa_config, name="Interactive")
	bf.simulate(s_id = 1,interactive=True, depth=90, update_rsa_prior=True)