from scipy.misc import imread
from enum import Enum
import matplotlib.image as mpimg
import sys
import numpy as np
from PIL import Image, ImageChops
import PIL
import scipy.misc

class ReadStates(Enum):
	READING_NONE			= 0,
	READING_LABELS 			= 1,
	READING_NAMES  			= 2,
	READING_TRAIN_INDEX		= 3,
	READING_TEST_INDEX		= 4,
	READING_VAL_INDEX		= 5,
	READING_TXN_INDEX		= 6,
	READING_CLASS_LABELS 		= 7,
	READING_CLASS_NAMES		= 8,

class EnglishFont:

	def __init__(self, filename):
		self.filename = filename
		self.images_list = []
		state = ReadStates.READING_NONE
		self.class_list = []
		self.training_matrix = []
		self.testing_matrix = []
		self.validation_matrix = []
		self.texton_matrix = []

		self.train_images = []
		self.test_images = []
		self.train_X = []
		self.train_Y = []

		with open(filename, "r") as f:
			for line in f:
				if line.strip() == "];":
					state = ReadStates.READING_NONE
					continue

				if line.startswith("list.ALLlabels"):
					state = ReadStates.READING_LABELS
					label = line.split("list.ALLlabels = [")[1]
					self.get_label(label)
					continue

				if line.startswith("list.ALLnames"):
					state = ReadStates.READING_NAMES
					name = line.split("list.ALLnames = [")[1].strip()
					self.get_full_name(name)
					continue

				if line.startswith("list.classlabels"):
					state = ReadStates.READING_CLASS_LABELS
					label = line.split("list.classlabels = [")[1]
					self.get_class_label(label)
					continue

				if line.startswith("list.classnames"):
					state = ReadStates.READING_CLASS_NAMES
					name = line.split("list.classnames = [")[1].strip()
					self.get_class_name(name)
					continue

				if line.startswith("list.NUMclasses"):
					parts = line.strip().split(";")
					self.num_classes = int(parts[0].split(" = ")[1])
					print("Num Classes: " + str(self.num_classes))

					if not parts[1].startswith("list.TRNind"):
						raise Exception("Invalid file format")

					state = ReadStates.READING_TRAIN_INDEX
					row = parts[1].split("list.TRNind = [")[1]
					self.add_training_row(row)
					continue

				if line.startswith("list.TSTind"):
					state = ReadStates.READING_TEST_INDEX
					row = line.split("list.TSTind = [")[1].strip()
					self.add_testing_row(row)
					continue

				if line.startswith("list.VALind"):
					state = ReadStates.READING_VAL_INDEX
					row = line.split("list.VALind = [")[1].strip()
					if row == '];':
						continue
					self.add_validation_row(row)
					continue

				if line.startswith("list.TXNind"):
					state = ReadStates.READING_TXN_INDEX
					row = line.split("list.TXNind = [")[1].strip()
					self.add_texton_row(row)
					continue

				if state == ReadStates.READING_TXN_INDEX:
					self.add_texton_row(line.strip())
					continue

				if state == ReadStates.READING_VAL_INDEX:
					self.add_validation_row(line.strip())
					continue

				if state == ReadStates.READING_TEST_INDEX:
					self.add_testing_row(line.strip())
					continue

				if state == ReadStates.READING_TRAIN_INDEX:
					self.add_training_row(line.split(";")[0])
					continue

				if state == ReadStates.READING_LABELS:
					self.get_label(line)
					continue

				if state == ReadStates.READING_CLASS_LABELS:
					self.get_class_label(line)
					continue

				if state == ReadStates.READING_NAMES:
					self.get_full_name(line.strip())
					continue

				if state == ReadStates.READING_CLASS_NAMES:
					self.get_class_name(line.strip())
					continue

		print("Number of images: " + str(len(self.images_list)))
		print("Number of classes: " + str(len(self.class_list)))
		print("Training rows: " + str(len(self.training_matrix)))
		print("Testing rows: " + str(len(self.testing_matrix)))
		print("Validation rows: " + str(len(self.validation_matrix)))
		print("Texton rows: " + str(len(self.texton_matrix)))

	# Returns the images for train, index must be between 1 to 30
	def get_train_images(self, index):
		if index < 1 or index > 30:
			raise Exception("Invalid index specified, needs index from 1 to 30")

		index -= 1
		train_images = []
		for index_list in self.training_matrix:
			image_index = index_list[index]
			if image_index == 0:
				continue
			image = self.images_list[image_index - 1]
			train_images.append(image)

		return train_images

	def get_test_images(self, index):
		if index < 1 or index > 30:
			raise Exception("Invalid index specified, needs index from 1 to 20")

		index -= 1
		test_images = []
		for index_list in self.testing_matrix:
			image_index = index_list[index]
			if image_index == 0:
				continue
			image = self.images_list[image_index - 1]
			test_images.append(image)

		return test_images

	def process(self):
		print("Processing image data at index 1: ")
		self.train_images = self.get_train_images(30)
		self.test_images = self.get_test_images(30)

	def add_training_row(self, row):
		index_list = row.strip().split(" ")
		self.training_matrix.append([int(x) for x in index_list])

	def add_testing_row(self, row):
		index_list = row.strip().split(";")[0].strip().split(" ")
		self.testing_matrix.append([int(x) for x in index_list])

	def add_validation_row(self, row):
		index_list = row.strip().split(";")[0].strip().split(" ")
		self.validation_matrix.append([int(x) for x in index_list])

	def add_texton_row(self, row):
		index_list = row.strip().split(";")[0].strip().split(" ")
		self.texton_matrix.append([int(x) for x in index_list])

	def get_full_name(self, name):
		name = name.replace("'", "")
		name = "./English/Hnd/" + name + ".png"
		entry = self.images_list.pop(0)
		entry['image'] = name
		self.images_list.append(entry)

	def get_class_name(self, name):
		name = name.replace("'", "")
		entry = self.class_list.pop(0)
		entry['class_name'] = name
		self.class_list.append(entry)

	def get_class_label(self, label):
		entry = {}
		label = int(label.strip().split(";")[0])
		entry = {'label': label}
		self.class_list.append(entry)

	def get_label(self, label):
		entry = {}
		label = int(label.strip().split(";")[0])
		entry = {'label': label}
		self.images_list.append(entry)

	def copy_all(self):
		f = open('hndall.txt', 'w')
		basewidth = 100

		for dict in self.images_list:
			s = 'r' + dict['image'][28:] + ',' + str(dict['label']) + '\n'
			f.write(s)
			img = Image.open(dict['image'])
			img = self.trim(img)
			img = img.resize((100, 100), PIL.Image.ANTIALIAS)
			new_img_path = './hndall/' + 'r' + dict['image'][28:]
			img.save(new_img_path)
		return

	def trim(self, im):
		bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
		diff = ImageChops.difference(im, bg)
		bbox = diff.getbbox()
		if bbox:
			return im.crop(bbox)


if __name__ == "__main__":
	ef = EnglishFont("list_English_Hnd.m")
	ef.process()
	ef.copy_all()
	print("END OF PROGRAM")

