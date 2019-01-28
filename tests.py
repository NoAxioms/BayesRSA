from utilities import bayes_rule, uniform, softmax
from rsaClass import RSA
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
def test_permutation_invariance():
	faces_classic = (("face",),("face","moustache"),("face","moustache","glasses"))
	permutations = list(itertools.permutations([i for i in range(len(faces_classic))]))
	rsa_list = []
	for perm in permutations:
		permuted_items = tuple((faces_classic[perm[i]] for i in range(len(perm))))
		print(permuted_items)
		rsa_list.append(rsa_test(permuted_items,depth=10)[0])
	speaker_probs_standard = rsa_list[0].speaker_probs
	for perm_id, perm in enumerate(permutations):
		#Compare to first permutation
		de_permuted_probs = np.empty(shape=(speaker_probs_standard.shape))
		permuted_probs = rsa_list[perm_id].speaker_probs
		print("\npermutation: {}".format(perm))
		print("standard_probs: \n{}".format(speaker_probs_standard))
		print("permuted_probs: \n{}".format(permuted_probs))
		for s_id, s in enumerate(perm):
			de_permuted_probs[s,:] = permuted_probs[s_id,:]

		assert np.array_equal(de_permuted_probs,speaker_probs_standard), "{} {}\n{}\n\n{}".format(perm_id,perm,speaker_probs_standard,de_permuted_probs)
def rsa_test(items, depth=1, priors = None, theta=5.0):
	print("theta: {}".format(theta))
	if priors is None:
		priors = (uniform(len(items)),)
	rsa_list = []
	for p in priors:
		rsa = RSA(items=items,theta=theta,listener_prior = p)
		# print("Vocab: {}".format(rsa.vocab))
		# print("Items: {}".format(items))
		# print("Prior: {}".format(p))
		# print("\nListener_0 [u][s]:\n{}".format(rsa.listener_probs))
		# print("\nSpeaker_0 [s][u]:\n{}".format(rsa.speaker_probs))
		print(rsa)
		rsa.run(depth=depth)
		print(rsa)
		# print("\nListener_{} [u][s]:\n{}".format(depth, rsa.listener_probs))
		# print("\nSpeaker_{} [s][u]:\n{}".format(depth, rsa.speaker_probs))
		rsa_list.append(rsa)
	return rsa_list
def rsa_classic_test():
	np.set_printoptions(precision=3,suppress=True)
	faces_classic = (("face",),("face","moustache"),("face","moustache","glasses"))
	faces_sym = (("face","moustache"), ("face","moustache","glasses"),("face", "glasses"))
	priors = (uniform(3),np.array((0.01,0.495,0.495)))
	rsa_test(faces_classic,depth=1000,theta=10.0,priors=priors)
rsa_classic_test()