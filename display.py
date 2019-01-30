import os, copy
import numpy as np
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


# def displayDistributionList(distribution_list,items, observations = None):
# 	"""	Displays a list of distributions. Navigate between distributions using left, right. 
# 	Displays observation on plot as well if observations is not None.
# 	If it becomes sluggish, rewrite to reuse previously created graphs
# 	for when a human is providing observations and the distribution list is created iteratively"""
# 	#Get figures
# 	def onKeyPress(event):
# 		global active_distribution
# 		if event.key == 'a':
# 			if active_distribution > 0:
# 					active_distribution	-= 1
# 		elif event.key == 'd':
# 			if active_distribution + 1	< len(items):
# 					active_distribution	+= 1
# 		# plt.show(fig_list[active_distribution])
# 	if observations is None:
# 		observations = ["N/A" for x in distribution_list]
# 	fig_list = []
# 	for b_id, b in enumerate(distribution_list):
# 		fig = displayDistribution(b,items,observations[b_id])
# 		# fig.canvas.mpl_connect('key_press_event', onKeyPress)
# 		fig_list.append(fig)
# 	plt.show(fig_list[active_distribution])


# class TrajectoryDisplay():
# 	def __init__(self, items, b_list = None, o_list = None):
# 		self.items = items
# 		self.b_list = copy.copy(b_list) if b_list is not None else []
# 		self.o_list = copy.copy(o_list) if o_list is not None else []
# 		self.fig_list = []
# 		self.active_distribution = 0
# 	def generateAllFigures(self):
# 		print("Generating all figures")
# 		self.fig_list = []
# 		for b_id, b in enumerate(self.b_list):
# 			o = self.o_list[b_id]
# 			print("{}\n{}\n{}".format(b,self.items,o))
# 			self.fig_list.append(self.generateFigure(b, self.items, o))
# 	def display(self):
# 		fig = self.fig_list[self.active_distribution]
# 		# for i in range(len(self.fig_list)):
# 			# plt.close()
# 			# pass
# 		plt.close()
# 		plt.show(fig)
# 	def generateFigure(self,distribution, items, observation, addOnKey = False):
# 		num_items = distribution.shape[0]
# 		# bar_width = 1.0/(1.5 * num_items)
# 		bar_width = 0.05
# 		locs = [float(i + 1.0)/(num_items+1)-0.05 for i in range(num_items)]
# 		fig, ax = plt.subplots()

# 		ax.set_xlim(left=0,right=1)
# 		ax.set_ylim(bottom=0,top=1)
# 		ax.xaxis.set_ticks_position("none")
# 		ax.yaxis.set_ticks_position("none")
# 		ax.spines['top'].set_visible(False)
# 		ax.spines['right'].set_visible(False)
# 		ax.xaxis.set_ticklabels(["" for i in ax.xaxis.get_ticklabels()])
# 		bar = ax.bar(locs,distribution,bar_width, align='edge')
# 		image_axis_height = 0.0
# 		image_x_offset = -0.028
# 		for i_id in range(num_items):
# 			image_path = item_to_image_path(items[i_id])
# 			img = mpimg.imread(image_path)
# 			bar_i = bar.patches[i_id]
# 			left = fig.transFigure.inverted().transform((ax.transAxes.transform((locs[i_id],image_axis_height))))[0] + image_x_offset
# 			rect = (left,image_axis_height, .1, .1)
# 			a_i = fig.add_axes(rect)
# 			a_i.axison=False
# 			imgplot	 = a_i.imshow(img)
# 		ax.axison=True
# 		# plt.show()
# 		if addOnKey:
# 			fig.canvas.mpl_connect('key_press_event', lambda event: self.onKeyPress(event))
# 		return fig
# 	def onKeyPress(self, event):
# 		previousKey = 'a'
# 		nextKey = 'd'
# 		if event.key == previousKey:
# 			if self.active_distribution > 0:
# 					self.active_distribution	-= 1
# 		elif event.key == nextKey:
# 			if self.active_distribution + 1	< len(self.items):
# 					self.active_distribution	+= 1
# 		if event.key in [previousKey, nextKey]:
# 			plt.show(self.fig_list[self.active_distribution])
# if __name__ == "__main__":
# 	items = (("face",),("face","moustache"),("face","moustache","glasses"))
# 	belief_list = [np.array([0.5,0.3,.2])]
# 	# belief = np.array([0.5,0.3])
# 	# displayDistribution(belief,items)
# 	# fig0 = displayDistribution(belief_list[0],items)
# 	# plt.show(fig0)
