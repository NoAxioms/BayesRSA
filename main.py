import json
from BayesClass import BayesFilter
from item_generator import generate_items
# from display import displayDistribution, TrajectoryDisplay
def experiment(num_trials = 10, trial_length = 1, rsa_config = {}):
	"""Currently even classic faces fail. This is almost certainly a bug."""
	with open("./standard_faces.json", 'r') as f:
		standard_faces = json.load(f)
		faces_classic = tuple(x.split(" ") for x in standard_faces["faces_classic"])
	faces = faces_classic
	# faces = generate_items(3)
	num_items = len(faces)
	rsa_config = {
		"theta":10.0,
		"default_depth":10
	}.update(rsa_config)
	for update_rule in [True, False]:
		print("update_rule: {}".format(update_rule))
		successes_per_item = [0] * num_items
		failures_per_item = [0] * num_items
		for i_id in range(num_items):
			print("want item {}: {}".format(i_id,faces[i_id]))
			for t in range(num_trials):
				bf = BayesFilter(items = faces, rsa_config = rsa_config)
				b_list, o_list = bf.simulate(i_id,trial_length, update_rsa_prior = update_rule, record = False, verbose = True)
				if b_list[-1][i_id] > 0.9:
					successes_per_item[i_id] += 1
				else:
					failures_per_item[i_id] += 1
		print("successes: {}".format(successes_per_item))
		print("failures:  {}".format(failures_per_item))
	# td = TrajectoryDisplay(faces,b_list,o_list)
	# td.generateAllFigures()
	# td.display()
if __name__ == "__main__":
	experiment()