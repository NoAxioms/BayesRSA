from utilities import bayes_rule
import numpy as np
def bayes_rule_test(a=None, b=None, a_cond_b = None, b_cond_a = None):
	if a is None:
		a = np.array([13. / 32, 19. / 32])
		b = np.array([1. / 4, 3. / 4])
		a_cond_b = np.array([
			[1. / 2, 1. / 2],
			[3. / 8, 5. / 8]
		])
		b_cond_a = np.array([
			[4./13,9./13],
			[4./19,15./19]
		])
	b_cond_a_derived = bayes_rule(b, a_cond_b)
	b_cond_a_derived_debug = bayes_rule(b,a_cond_b,a)
	assert np.max(np.abs(b_cond_a_derived - b_cond_a_derived_debug)) < 0.0001, "\n{}\n{}".format(b_cond_a_derived, b_cond_a_derived_debug)
	delta = b_cond_a - b_cond_a_derived
	max_abs_delta = np.max(np.abs(delta))
	assert max_abs_delta < 0.0001, "\n{}\n{}".format(b_cond_a, b_cond_a_derived)
	a_cond_b_derived = bayes_rule(a,b_cond_a)
	delta2 = a_cond_b - a_cond_b_derived
	max_abs_delta2 = np.max(np.abs(delta2))
	assert max_abs_delta2 < 0.0001, "\n{}\n{}".format(a_cond_b, a_cond_b_derived)
bayes_rule_test()