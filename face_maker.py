import math, copy, os
from lxml import etree
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
from utilities import powerset
# from cairosvg import svg2png
# from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPDF, renderPM

components_all = ["face","moustache","glasses","hat","ears","nose","eyes","eyebrows","hair","beard","teeth"]
face_vector_directory = os.path.dirname(os.path.realpath(__file__)) + "/face_vectors"
face_image_directory = os.path.dirname(os.path.realpath(__file__)) + "/face_images/"
def makeFace(components):
	components = sorted(components)
	tree = etree.parse(open(face_vector_directory + "face_all.svg",'r'))
	missing_names = [c for c in components_all if c not in components]
	missing_elements = []
	for element in tree.iter():
		if element.get("id") in missing_names:
			missing_elements.append(element)
	for element in missing_elements:
		element.getparent().remove(element)
	new_name = "_".join(components)
	tree.write(face_vector_directory + new_name + ".svg")
	drawing = svg2rlg(face_vector_directory + new_name + ".svg")
	# renderPDF.drawToFile(drawing, "file.pdf")
	renderPM.drawToFile(drawing, face_image_directory + new_name + ".png", fmt="PNG")
def item_to_image_path(item):
	new_name = "_".join(item) + ".png"
	return face_image_directory + new_name
if __name__ == "__main__":
	faces = powerset(["face","moustache","glasses","hat"])[0:-1]
	# faces_classic = (("face",),("face","moustache"),("face","moustache","glasses"))
	for c in faces:
		makeFace(c)