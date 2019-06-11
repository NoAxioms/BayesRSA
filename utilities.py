import torch

def global_2_local_id(global_id, ommited_ids):
	num_ommited_predecessors = len([i for i in ommited_ids if i < global_id])
	return global_id - num_ommited_predecessors

def copy_array_sans_indices(old_array, skipped_indices):
	kept_indices = [i for i in range(old_array.shape[0]) if i not in skipped_indices]
	new_array = old_array[kept_indices].clone().detach()
	return new_array


def normalize(x):
	"""
	Normalizes 2d tensor so that x.sum(1)[i] = 1.0
	TODO make dim argument
	TODO make work with broadcasting for enumeration stuff
	"""
	assert len(x.shape) == 2, x.shape
	d = x.sum(dim=1).repeat(x.shape[1]).reshape(
		x.shape[1], x.shape[0]).transpose(0, 1)
	assert d.shape == x.shape, "x.shape: {};    d.shape: {}".format(
		x.shape, d.shape)
	return x / d


def softmax(x, theta=1.0):
	# 2d only
	exponentiated = torch.exp(theta * x)
	return normalize(exponentiated)

def bayes_rule(b, a_cond_b, a=None, debug=False):
	"""
	:param a: Prob(a) matrix
	:param b: Prob(b) matrix
	:param a_cond_b: prob(a | b) matrix with [b][a] indexing
	:return: prob(b | a) matrix with [b][a] indexing
	"""
	# Calculate a from b and a_cond_b if a=None
	# normalize
	a_cond_b = normalize(a_cond_b)
	b = b / b.sum()
	if a is None:
		a = torch.einsum('b,ba->a', b, a_cond_b)
		if debug:
			print("a:\n", a)
	# b_stretched = b.repeat(a.shape[0]).reshape((b.shape[0],a.shape[0])).transpose(0,1)  #[b][a]
	b_stretched = b.reshape((b.shape[0], 1)).repeat(1, a.shape[0])
	if debug:
		print("a_cond_b:\n", a_cond_b)
		print("b_stretched:\n", b_stretched)
	a_join_b = (a_cond_b * b_stretched)  # [b][a]
	if debug:
		print("a_join_b:\n", a_join_b)
	a_stretched = a.reshape(1, a.shape[0]).repeat(b.shape[0], 1)  # [b][a]
	if debug:
		print("a_stretched:\n", a_stretched)
	b_cond_a = (a_join_b / a_stretched)  # [b][a]
	# If some values of a have prob 0, we get nans. Replace nans with 0.
	nan_id = torch.isnan(b_cond_a)
	b_cond_a[nan_id] = 0.0
	if debug:
		print("b_cond_a:\n", b_cond_a)
		print("nan_id:\n", nan_id)
	return b_cond_a


def l1_distance(a, b):
	return torch.abs(a - b).sum()

def cart2sph(cart):
	r = torch.norm(cart,2)
	az = torch.atan2(cart[1],cart[0])
	hxy = torch.norm(cart[0:2],2)
	el = torch.atan2(cart[2],hxy)
	return torch.tensor([az,el,r])

def sph2cart(sph):
	rcos_theta = sph[2] * torch.cos(sph[1])
	x = rcos_theta * torch.cos(sph[0])
	y = rcos_theta * torch.sin(sph[0])
	z = sph[2] * torch.sin(1)
	return torch.tensor([x,y,z])