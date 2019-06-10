def run_trials():
	for trial_num in range(trials):
		observations = []
		while in_trial:
			obs = get_observations()
			#Handle novel words
			if not obs.words subset lexicon:
				lexicon.incorporate(obs.words)
				#Needed to infer that new word probably applies to previously unmentioned item. 
				#This feels like an argument to merge lexicon and target inference, since gesture informs which of the items have this new word.
				update_lexicon() 
			observations.append(obs)
			infer_target(observations)  #Does not affect lexicon? N
			act(belief_about_target)  #Perhaps split into known and believed items, similar to lexicon
			#If we somehow obtain knowledge, incorporate it
		if picked_correct:
			update_lexicon(...)

#Need to deal with the conflict between how models act for inference and how they can act to return valid samples.


def model():
	target = pyro.sample('target', categorical(pyro.param('target')) obs=target)
	language = language_model(target)

def guide():
	if target is not None:
		target = pyro.sample('target', categorical(pyro.param('target')))
	language = language_guide(target) #Does this make sense?