import os
from collections import namedtuple
import torch
import pyro
import pyro.distributions as dist
from pyro import ParamStore
from torch.distributions import constraints
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.optim import Adam


# from rsaClass import RSA
# from utilities import softmax
smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('0.3.1')
pyro.enable_validation()
pyro.set_rng_seed(0)

Revelation = namedtuple('Revelation', ['item','words', 'values'])
def global_2_local_id(global_id, ommited_ids):
	num_ommited_predecessors = len([i for i in ommited_ids if i < global_id])
	return global_id - num_ommited_predecessors

def copy_array_sans_indices(old_array, skipped_indices):
	kept_indices = [i for i in range(old_array.shape[0]) if i not in skipped_indices]
	new_array = old_array[kept_indices].clone().detach()
	return new_array

def copy_array_sans_indices_test():
	old_array = torch.arange(5)
	skipped_indices = [1,3,4]
	new_array = copy_array_sans_indices(old_array,skipped_indices)
	print(old_array)
	print(new_array)

def model(num_items, num_words, known_words_by_item, known_values_by_item):
	item_plate = pyro.plate('item_plate', num_items)
	s_0 = torch.empty((num_items,num_words))
	for i in item_plate:
		s_0[i][known_words_by_item[i]] = torch.tensor(known_values_by_item[i])
		unkown_words = [w for w in range(num_words) if w not in known_words_by_item[i]]
		s_0[i][unkown_words] = pyro.param('vocab_believed_{}'.format(i), torch.ones(len(unkown_words)) * 0.5)
	print(s_0)

def main():
	pyro.clear_param_store()
	num_items = 2
	num_words = 3
	#This kind of parallel structure feels unstable
	#Indices of known words
	known_words_by_item = [[] for _ in range(num_items)]
	#Values of known words. Synced to known_words_by_item
	known_values_by_item = [[] for _ in range(num_items)]

	adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
	optimizer = Adam(adam_params)
	svi = SVI(model, model, optimizer, loss=Trace_ELBO())
	svi.step(num_items=num_items, num_words=num_words, known_words_by_item=known_words_by_item, known_values_by_item=known_values_by_item)
	revelations = [Revelation(0,[0,1],[1,0])] #word 0 applied to item 0, word 1 does not
	#This is ugly. Would be pretty if each revelation were an item, word, value triple, 
	#but since the param we are updating is an array, I preferred this.
	for r in revelations:
		#Update knowledge
		redundant_words = []
		new_words_global = []
		for w_id_in_revelation, w in enumerate(r.words):
			if w not in known_words_by_item[r.item]:
				new_words_global.append(w)
			#If we know the value of the word
			else:
				w_index_in_known = known_words_by_item[r.item].index(w)
				current_val = known_values_by_item[r.item][w_index_in_known]
				new_val = r.values[w_id_in_revelation]
				if current_val != new_val:
					print("Certain knowledge was wrong: {}[{}]: {} -> {}".format(r.item,w,current_val,new_val))
					known_values_by_item[r.item][w_index_in_known] = new_val
		#Update belief param
		vocab_believed = pyro.param('vocab_believed_{}'.format(r.item), torch.ones(num_words) * 0.5)
		new_words_local = [global_2_local_id(w, known_words_by_item[r.item]) for w in new_words_global]
		vocab_believed_new = copy_array_sans_indices(vocab_believed,new_words_local)
		ParamStore.replace_param('vocab_believed_{}'.format(r.item),vocab_believed,vocab_believed_new)

	svi.step(num_items=num_items, num_words=num_words, known_words_by_item=known_words_by_item, known_values_by_item=known_values_by_item)

	# known_words_by_item[0].append(0)
	# pyro.param_store.replace_param("words_believed_0", torch)
	# svi.step(num_items=num_items, num_words=num_words, known_words_by_item=known_words_by_item)
main()