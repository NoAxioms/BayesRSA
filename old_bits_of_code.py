def enumeration_time_test():
	"""
	Compares speed of inference using enumeration v not using enumeration
	"""
	num_trials = 2
	"""
	The order of running seems to affect outcome - the second is faster, especially when the second is enumeration. Problem with globals?
	Possibilities:
	global/param not being reset
	Amortization under the hood

	"""
	sans_enumeration_beliefs = run_trials(use_enumeration=False, num_trials=num_trials, verbose=False, stop_method = "convergence")
	enumeration_beliefs = run_trials(use_enumeration = True, num_trials = num_trials, verbose=False, stop_method = "convergence")

	# enumeration_beliefs = run_trials(use_enumeration = True, num_trials = num_trials)
	# sans_enumeration_beliefs = run_trials(use_enumeration=False, num_trials=num_trials)

	for t in range(num_trials):
		print("Enumeration beliefs for trial {}. Num_steps: {}".format(t,len(enumeration_beliefs[t])))
		for step_num, d in enumerate(enumeration_beliefs[t]):
			print(d["target_item"])
		print("sans enumeration beliefs for trial {}. Num_steps: {}".format(t,len(sans_enumeration_beliefs[t])))

		for step_num, d in enumerate(sans_enumeration_beliefs[t]):
			print(d["target_item"])