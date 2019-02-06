import os, copy
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab as pl
from face_maker import item_to_image_path
image_directory = os.path.dirname(os.path.realpath(__file__)) + "/images/"
# img=mpimg.imread(image_directory + "face.png")
# active_distribution=0
#TODO RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory.
def createBeliefFigure(belief, items, observation = "N/A", save_location = None, display_now = False, des_id = None):
	# plt.tick_params(
	#     axis='x',          # changes apply to the x-axis
	#     which='both',      # both major and minor ticks are affected
	#     bottom=False,      # ticks along the bottom edge are off
	#     top=False,         # ticks along the top edge are off
	#     labelbottom=False) # labels along the bottom edge are off
	num_items = belief.shape[0]
	# bar_width = 1.0/(1.5 * num_items)
	bar_width = 0.05
	locs = [float(i + 1.0)/(num_items+1)-0.05 for i in range(num_items)]
	fig, ax = plt.subplots()

	ax.set_xlim(left=0,right=1)
	ax.set_ylim(bottom=0,top=1)
	ax.xaxis.set_ticks_position("none")
	ax.yaxis.set_ticks_position("none")
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.xaxis.set_ticklabels(["" for i in ax.xaxis.get_ticklabels()])
	bar = ax.bar(locs,belief,bar_width, align='edge')
	image_axis_height = 0.0
	image_x_offset = -0.028
	for i_id in range(num_items):
		image_path = item_to_image_path(items[i_id])
		img = mpimg.imread(image_path)
		bar_i = bar.patches[i_id]
		left = fig.transFigure.inverted().transform((ax.transAxes.transform((locs[i_id],image_axis_height))))[0] + image_x_offset
		rect = (left,image_axis_height, .1, .1)
		a_i = fig.add_axes(rect)
		a_i.axison=False
		imgplot	 = a_i.imshow(img)
	ax.axison=True
	plt.suptitle(observation)
	if des_id is not None:
		bar.patches[des_id].set_color('r')
	if display_now:
		plt.show()
	if save_location is not None:
		plt.savefig(save_location)
	plt.close()
	# plt.clf()
	# return fig

def plotBeliefSimplex(belief_list, values, value_scale = 2, start_color = (0.0,0.0,1.0), end_color = (1.0,0.0,0.0), title = "", axis_labels = ("","")):
	"""
	Currently only works for 3 dimensional belief
	Change to use images
	"""
	unzipped_beliefs = list(zip(*belief_list))
	print(type(unzipped_beliefs[0]))
	print(type(belief_list))
	print(type(belief_list[0]))
	#Fit values to color scheme
	colors = []
	for v_id in range(values.shape[0]):
		t = expit(values[v_id]/value_scale)
		# print("t: {}".format(t))
		c = [start_color[i] * t + end_color[i] * (1.0 - t) for i in range(3)] + [0.9]
		# print(c)
		assert np.max(c) <= 1.0
		assert np.min(c) >= 0
		colors.append(c)
	colors = np.array(colors)
	#Plot
	fig, ax = plt.subplots()
	scat = plt.scatter(list(unzipped_beliefs[0]),list(unzipped_beliefs[1]), c=colors)
	plt.suptitle(title)
	plt.xlabel(axis_labels[0])
	plt.ylabel(axis_labels[1])
	plt.show()

