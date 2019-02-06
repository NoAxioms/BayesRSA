import numpy as np
def softmax(X, theta=1.0, axis=None):
	"""
	Compute the softmax of each element along an axis of X.

	Parameters
	----------
	X: ND-Array. Probably should be floats.
	theta (optional): float parameter, used as a multiplier
		prior to exponentiation. Default = 1.0
	axis (optional): axis to compute values along. Default is the
		first non-singleton axis.

	Returns an array the same size as X. The result will sum to 1
	along the specified axis.
	"""

	# make X at least 2d
	y = np.atleast_2d(X)

	# find axis
	if axis is None:
		axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

	# multiply y against the theta parameter,
	y = y * float(theta)

	# subtract the max for numerical stability
	y = y - np.expand_dims(np.max(y, axis=axis), axis)

	# exponentiate y
	y = np.exp(y)

	# take the sum along the specified axis
	ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

	# finally: divide elementwise
	p = y / ax_sum

	# flatten if X was 1D
	if len(X.shape) == 1: p = p.flatten()

	return p

def bayes_rule(b,a_cond_b, a=None, debug = False):
	"""
	:param a: Prob(a) matrix
	:param b: Prob(b) matrix
	:param a_cond_b: prob(a | b) matrix with [b][a] indexing
	:return: prob(b | a) matrix with [a][b] indexing
	"""
	#Calculate a from b and a_cond_b if a=None
	if type(b) is list:
		b = np.array(b)
	if a is None:
		a = np.einsum('b,ba->a',b,a_cond_b)
	b_stretched = b.repeat(a.shape[0]).reshape((b.shape[0],a.shape[0]))  #[b][a]
	a_join_b = a_cond_b * b_stretched #[b][a]
	a_stretched = a.repeat(b.shape[0]).reshape((a.shape[0],b.shape[0])).swapaxes(0,1) #[b][a]
	b_cond_a = np.true_divide(a_join_b,a_stretched).swapaxes(0,1)
	return b_cond_a

def uniform(n):
	return np.array([1.0 / n for i in range(n)])
def arr2tex(a):
	return " \\\\\n".join([" & ".join(map(str,line)) for line in a])
def powerset(a):
	subsets = []
	for i in range(2**len(a)):
		current_subset = [a[j] for j in range(len(a)) if (i % (2**(j+1))) - (i %(2**j)) == 0]
		subsets.append(current_subset)
	return subsets
def freudenthal(depth, upper_bound, previous_segments=None, normalize = True):
	"""
	Taken from Lovejoy - Computational Feasible Bounds for Partially Observed Markov Decision Processes
	Section 4.2
	I don't remember what upper_bound does
	"""
	if previous_segments == None:
		extended_segments = [[upper_bound]]
	else:
		extended_segments = []
		for segment in previous_segments:
			for i in range(segment[-1] + 1):
				extended_segments.append(segment + [i])
	if depth > 1:
		extended_segments = freudenthal(depth - 1, upper_bound, extended_segments)
	broken_segments = [g for g in extended_segments if g[0] != upper_bound]
	if len(broken_segments) > 0:
		print(broken_segments)
		raise ValueError("Freudenthal is busted")
	return extended_segments
def get_belief_grid(num_states, resolution):
	grid_large = freudenthal(num_states, resolution)
	upper_bound = float(resolution)
	# grid_normal = []
	# for g in grid_large:
	# 	g_norm = []
	# 	for i in range(len(g)-1):
	# 		g_norm.append(g[i] - g[i+1])
	# 	g_norm.append(g[-1]/u)
	grid_normal = [[(g[i] - g[i + 1]) / upper_bound for i in range(num_states - 1)] + [g[-1] / upper_bound] for g
	               in grid_large]
	sizes = list(set([len(g) for g in grid_normal]))
	if len(sizes) > 1 or len(grid_normal[0]) != num_states:
		raise ValueError(
			"Grid has arrays of " + str(len(sizes)) + " lengths and first g has length " + str(len(grid_normal[0])))
	null_indices = [i for i in range(len(grid_normal)) if
	                grid_normal[i] == [0 for j in range(num_states)]]
	null_larges = [grid_large[i] for i in null_indices]
	if len(null_larges) > 0:
		raise ValueError("have beliefs of all 0")
	print("null beliefs: " + str(len(null_larges)))
	for i in range(min(len(null_larges), 10)):
		print(null_larges[i])

	errors = [abs(1 - sum(g)) for g in grid_normal]
	max_error = max(errors)
	worst_g = max(grid_normal, key=lambda a: abs(1 - sum(a)))
	if max(errors) > 0.01:
		print("worst g: ")
		print(worst_g)
		raise ValueError("Fruedenthal created belief with error " + str(max(errors)))
	return grid_normal

def list_depth(l):
	"""Empty list returns 1"""
	d = 0
	if len(l) == 0:
		return 1
	running = True
	while running:
		try:
			l = l[0]
			d += 1
		except:
			running = False
	return d

