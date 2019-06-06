import torch
import pyro
from torch.distributions import constraints, transform_to



param_store = pyro.get_param_store()

def delete_test():
	a = pyro.param('a', torch.ones(3))
	print(param_store.keys())  #dict_keys(['a'])
	# print(param_store._param_to_name.keys())
	param_store.__delitem__('a')  #Works fine
	print(param_store.keys())  #dict_keys([])
	# print(param_store._param_to_name.keys())
	# b = pyro.param('b', torch.ones(2) * .5, constraint=constraints.positive)
	b = param_store.setdefault('b', torch.ones(2) * .5, constraint=constraints.positive)
	print(param_store.keys())  #dict_keys(['b'])
	print(pyro.param('b'))
	# print(param_store._param_to_name.keys())
	# print(param_store._params.values())
	# print("deleting b")
	param_store.__delitem__('b')  #AttributeError: 'Tensor' object has no attribute 'unconstrained'
	print(param_store.keys())
def set_test():
	b = pyro.param('b', torch.ones(2) * .5, constraint=constraints.positive)
	print(param_store.keys())
	print(param_store._param_to_name.keys())
	param_store.__setitem__('b', torch.ones(2))
	print(param_store.keys())
	print(param_store._param_to_name.keys())

def set_v_del_test():
	b = pyro.param('b', torch.ones(2) * .5, constraint=constraints.positive)
	print(b)
	# store constraint, defaulting to unconstrained
	constraint = param_store._constraints.setdefault('b', constraints.real)
	print(constraint)
	# compute the unconstrained value
	with torch.no_grad():
		# FIXME should we .detach() the new_constrained_value?
		unconstrained_value = transform_to(constraint).inv(b)
		unconstrained_value = unconstrained_value.contiguous()
	unconstrained_value.requires_grad_(True)
	print(unconstrained_value)
	print(param_store._param_to_name.keys())
	# param_store.__setitem__('b', torch.ones(2))
delete_test()