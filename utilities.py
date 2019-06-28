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

#TODO broadcast these
def l1_distance(a, b):
	return torch.abs(a - b).sum()



def cart2sph(cart):
	"""
	:param cart: either a 1d tensor of [x,y,z], or a 2d tensor of [[x0,y0,z0],[x1,y1,z1],...]
	:return: Either a 1d tensor of [az,el,r] or a 2d tensor of [[az0,el0,r0],[az1,el1,r1],...]
	use negative indices, ellipses notation bc last indices are event indices.
	"""
	#TODO make more memory efficient by assigning az, el, r directly to sph_flat
	#List of (x,y,z) triples
	cart_flat = cart.view(-1,3)
	# print("cart: {}".format(cart))
	r = torch.norm(cart_flat,2, dim=-1)
	#TODO vectorize arctan2
	#Option: flatten to 2d (-1 is (x,y,z)). For i in range(shape[0]):
	#each (x,y,z) triple gets a single az
	hxy = torch.norm(cart_flat[...,0:2],2, dim=-1)
	# print("hxy: {}".format(hxy))
	az = torch.empty(cart_flat.shape[0])
	el = torch.empty(cart_flat.shape[0])
	#Iterate through. 
	for i in range(cart_flat.shape[0]):
		torch.atan2(cart_flat[i][1],cart_flat[i][0],out=az[i])
		torch.atan2(cart_flat[i][2],hxy[i], out=el[i])
	# print("az, el, r: {}, {}, {}".format(az,el,r))
	sph_flat = torch.empty(cart_flat.shape)
	sph_flat[:,0] = az
	sph_flat[:,1] = el
	sph_flat[:,2] = r
	sph = sph_flat.view(cart.shape)

	return sph

def cart2sph_legacy(cart):
	"""
	:param cart: either a 1d tensor of [x,y,z], or a 2d tensor of [[x0,y0,z0],[x1,y1,z1],...]
	:return: Either a 1d tensor of [az,el,r] or a 2d tensor of [[az0,el0,r0],[az1,el1,r1],...]
	use negative indices, ellipses notation bc last indices are event indices.
	"""
	r = torch.norm(cart,2)
	az = torch.atan2(cart[1],cart[0])
	hxy = torch.norm(cart[0:2],2)
	el = torch.atan2(cart[2],hxy)
	# print("az, el, r: {}, {}, {}".format(az,el,r))
	return torch.tensor([az,el,r])

def sph2cart(sph):
	"""
	:param sph: eiter 1d or nd tensor
	"""
	sph_flat = sph.view(-1,3)
	cart_flat = torch.empty(sph_flat.shape)
	r_flat = sph_flat[:,2]
	cos_theta_flat = torch.cos(sph_flat[:,1])
	assert r_flat.shape == cos_theta_flat.shape, "{}, {}".format(r_flat.shape, cos_theta_flat.shape)
	rcos_theta_flat = r_flat * cos_theta_flat  #If this function doesnt work, check that these have the same shape
	cart_flat[:,0] = rcos_theta_flat * torch.cos(sph_flat[:,0])
	cart_flat[:,1] = rcos_theta_flat * torch.sin(sph_flat[:,0])
	cart_flat[:,2] = sph_flat[:,2] * torch.sin(sph_flat[:,1])
	cart = cart_flat.view(sph.shape)
	return cart

def sph2cart_legacy(sph):
	rcos_theta = sph[2] * torch.cos(sph[1])
	x = rcos_theta * torch.cos(sph[0])
	y = rcos_theta * torch.sin(sph[0])
	z = sph[2] * torch.sin(sph[1])
	return torch.tensor([x,y,z])

def tensor_index(tensor, values):
	"""
	Assuming each item in tensor appears once in values, returns a tensor of the indices
	"""
	if type(tensor) is int:
		tensor = torch.tensor([tensor])
	if tensor.shape == torch.Size([]):
		print('car')
		tensor = torch.tensor([tensor.item()])
	# print(tensor.shape)
	# print(values.shape)
	# print(torch.tensor([2]))
	c =  torch.nonzero(tensor[..., None] == values)
	# print("tensor_index")
	# print("{}\n{}\n{}".format(tensor,values,c))
	return c[:,1]

if __name__ == "__main__":
	pass