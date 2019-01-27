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