from utilities import powerset
def generate_items(num_words = 2):
	vocab = sorted(["moustache","glasses","hat","ears","nose","eyes","eyebrows","hair","beard","teeth"][0:num_words])
	#Get a face with any subset of the next num_words features
	items = sorted([tuple(["face"] + x) for x in powerset(vocab)])
	# items = [tuple(x) for x in powerset(vocab)]
	items = tuple(items)
	return items
if __name__ == "__main__":
	items = generate_items()
	print(items)
