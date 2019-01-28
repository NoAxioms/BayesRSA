import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab as pl
from face_maker import item_to_image_path
image_directory = os.path.dirname(os.path.realpath(__file__)) + "/images/"
# img=mpimg.imread(image_directory + "face.png")

def displayDistribution(distribution, items):
	num_items = distribution.shape[0]
	# bar_width = 1.0/(1.5 * num_items)
	bar_width = 0.05
	locs = [float(i + 1.0)/(num_items+1)-0.05 for i in range(num_items)]
	fig, ax = plt.subplots()
	ax.set_xlim(left=0,right=1)
	a=ax.get_xaxis()
	# print(a)
	# print("figure width: {}".format(fig.patch.get_width()))
	bar = ax.bar(locs,distribution,bar_width, align='edge')
	image_axis_height = 0.0
	image_x_offset = -0.022
	# image_axis_leftmost = 0.4
	# image_increment = 0.8 / num_items
	# bar_transform = bar.patches.get_patch_transform()
	for i_id in range(num_items):
		image_path = item_to_image_path(items[i_id])
		img = mpimg.imread(image_path)
		bar_i = bar.patches[i_id]
		# left = locs[i_id]
		left = fig.transFigure.inverted().transform((ax.transAxes.transform((locs[i_id],image_axis_height))))[0] + image_x_offset
		# left = (bar_i.get_x() + bar_i.get_width()/2)/fig.patch.get_width()
		# print("{}:  {}".format(i_id, left))
		rect = (left,image_axis_height, .1, .1)
		a_i = fig.add_axes(rect)
		a_i.axison=False
		imgplot	 = a_i.imshow(img)
	ax.axison=True
	plt.show()
if __name__ == "__main__":
	# A = np.arange(25).reshape(5,5)
	# fig, ax = plt.subplots(1, 1)

	# xl, yl, xh, yh=np.array(ax.get_position()).ravel()
	# w=xh-xl
	# h=yh-yl
	# xp=xl+w*0.1 #if replace '0' label, can also be calculated systematically using xlim()
	# size=0.05

	# # img=mpimg.imread('microblog.png')
	# ax.matshow(A)
	# ax1=fig.add_axes([xp-size*0.5, yh * 0.5, size, size])
	# ax1.axison = True
	# imgplot = ax1.imshow(img,transform=ax.transAxes)

	# plt.savefig('temp.png')
	# items = (("face",), ("face","glasses"), ("face","moustache"))
	items = (("face","moustache"),("face","moustache","glasses"))
	# items = (("face",),("face","moustache"),("face","moustache","glasses"))
	# belief = np.array([0.5,0.3,.2])
	belief = np.array([0.5,0.3])
	displayDistribution(belief,items)