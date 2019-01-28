from BayesClass import BayesFilter
from item_generator import generate_items
from display import displayDistribution

def main():
	faces = generate_items(3)
	rsa_kwargs = {
		"theta":20.0,
		"default_depth":10
	}
	bf = BayesFilter(items = faces, **rsa_kwargs)

	bf.simulate(3,10)

main()