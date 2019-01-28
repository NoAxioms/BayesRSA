from utilities import powerset
def generate_items(num_words = 3):
	vocab = ["moustache","glasses","ears","nose","eyes","eyebrows","hair","beard","teeth"][0:num_words]
	#Get a face with any subset of the next three features
	items = [tuple(["face"] + x) for x in powerset(vocab)]
	# items = [tuple(x) for x in powerset(vocab)]
	items = tuple(items)
	return items
if __name__ == "__main__":
	items = generate_items()
	print(items)
