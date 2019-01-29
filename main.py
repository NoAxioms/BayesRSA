from BayesClass import BayesFilter
from item_generator import generate_items
# from display import displayDistribution, TrajectoryDisplay

def main():
	faces = generate_items(3)
	rsa_config = {
		"theta":20.0,
		"default_depth":10
	}
	bf = BayesFilter(items = faces, rsa_config = rsa_config, name="dummy2")

	b_list, o_list = bf.simulate(3,10, update_rsa_prior = False)
	# td = TrajectoryDisplay(faces,b_list,o_list)
	# td.generateAllFigures()
	# td.display()
main()