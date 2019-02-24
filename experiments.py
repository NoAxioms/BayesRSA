import json, os, sys
import numpy as np
from rsaClass import RSA
np.set_printoptions(precision=2,floatmode='fixed')
def word_clearing_test():
	with open("./standard_faces.json", 'r') as f:
		standard_faces = json.load(f)
		faces = tuple(x.split(" ") for x in standard_faces["faces_sequence_alph"])
	rsa = RSA(items=faces, theta=5)
	for i in range(10):
		print("Depth {}".format(i))
		for w_id, w in enumerate(rsa.vocab):
			# print("{}: {}".format(w,rsa.speaker_probs[:,w_id]))
			print("{}: {}".format(w,rsa.listener_probs[w_id]))
		rsa.run(depth=1)

if __name__ == "__main__":
	word_clearing_test()