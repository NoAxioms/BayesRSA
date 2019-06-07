import torch

def global_2_local_id(global_id, ommited_ids):
	num_ommited_predecessors = len([i for i in ommited_ids if i < global_id])
	return global_id - num_ommited_predecessors
def copy_array_sans_indices(old_array, skipped_indices):
	kept_indices = [i for i in range(old_array.shape[0]) if i not in skipped_indices]
	new_array = old_array[kept_indices].clone().detach()
	return new_array