import json, os, sys
from BayesClass import BayesFilter
from item_generator import generate_items
from display import createBeliefFigure
from rsaClass import RSAExplorer
try:
	input
except:
	input = raw_input
# from display import displayDistribution, TrajectoryDisplay
def experiment(num_trials = 10, trial_length = 10, rsa_config = None):
	"""Currently even classic faces fail. This is almost certainly a bug."""
	with open("./standard_faces.json", 'r') as f:
		standard_faces = json.load(f)
		faces_classic = tuple(x.split(" ") for x in standard_faces["faces_classic"])
		faces_sym = tuple(x.split(" ") for x in standard_faces["faces_sym"])
	# faces = faces_classic
	faces = generate_items(3)
	# faces = faces_sym
	print("Faces: {}".format(faces))
	num_items = len(faces)
	if rsa_config is None: 
		rsa_config = {
			"theta":10.0,
			"default_depth":10
		}
	bf = BayesFilter(items = faces, rsa_config = rsa_config)
	trajectories_dict = {} #[update_type][des_id][trial_num][step_num] = (b,o)
	for update_rule in [True, False]:
		print("update_rule: {}".format(update_rule))
		successes_per_item = [0] * num_items
		failures_per_item = [0] * num_items
		trajectories_dict[update_rule] = [] 
		for des_id in range(num_items):
			# print("want item {}: {}".format(des_id,faces[des_id]))
			trajectories_for_item = []
			trajectories_dict[update_rule].append(trajectories_for_item)
			for t in range(num_trials):
				b_list, o_list = bf.simulate(des_id,trial_length, update_rsa_prior = update_rule, save_images = False, save_trajectories = True)
				trajectories_for_item.append(list(zip(b_list,o_list)))
				if b_list[-1][des_id] > 0.9:
					successes_per_item[des_id] += 1
				else:
					failures_per_item[des_id] += 1
		print("successes: {}".format(successes_per_item))
		print("failures:  {}".format(failures_per_item))
	record = input("Record? (y/n): ")
	if record == 'y':
		if not os.path.isdir(bf.record_path): os.mkdir(bf.record_path)
		for update_rule in [True,False]:
			for des_id in range(num_items):
				subdir_name = "rsa_uses_belief_{}_face_{}/".format(update_rule,des_id)
				os.mkdir(bf.record_path + subdir_name)
				for trial_num in range(num_trials):
					for step_num in range(trial_length):
						file_name = "trial_{}_step_{}.png".format(trial_num,step_num)
						file_path = bf.record_path + subdir_name + file_name
						b, o = trajectories_dict[update_rule][des_id][trial_num][step_num]
						createBeliefFigure(b, faces, o, save_location=file_path, des_id = des_id)
	# td = TrajectoryDisplay(faces,b_list,o_list)
	# td.generateAllFigures()
	# td.display()
if __name__ == "__main__":
	with open("./standard_faces.json", 'r') as f:
		standard_faces = json.load(f)
		faces_classic = tuple(x.split(" ") for x in standard_faces["faces_classic"])
		faces_sym = tuple(x.split(" ") for x in standard_faces["faces_sym"])
	rsaExplorer = RSAExplorer(items=faces_classic, theta=5.0)
