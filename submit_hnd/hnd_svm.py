import sys
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
from skimage.io import imread
from scipy import ndimage
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
import seaborn as sns
import matplotlib.pyplot

class HND_SVM:
	def __init__(self, filename):
		self.filename = filename
		self.train_X = []
		self.train_Y = []
		self.test_X = []
		self.test_VAL = []
		self.test_Y = []
		self.all_X = []
		self.all_Y = []
		self.model = None

	def read_all(self):
		print("read_all()")
		fo = open(self.filename, 'r')
		i = 0

		for line in fo:
			lst = line.split(',')
			path = './hndall/' + lst[0]
			img = imread(path, as_grey = True)

			fd, hog_image = hog(img, orientations=10, pixels_per_cell = (14,14), cells_per_block = (1,1), visualise = True)
			self.all_X.append(fd)
			self.all_Y.append(lst[1])

	def split_data(self):
		print("train_test_split()")
		self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(self.all_X, self.all_Y, test_size = 0.3)
		print("train_X length: " + str(len(self.train_X)))
		print("train_Y length: " + str(len(self.train_Y)))
		print("test_X length: " + str(len(self.test_X)))
		print("test_Y length: " + str(len(self.test_Y)))

	def train_classifier(self):
		print("train the classifier now")

		self.train_X = preprocessing.normalize(self.train_X)

		clf = SVC(kernel = 'linear', decision_function_shape = 'ovo') # 80%
#$		clf = SVC(kernel = 'linear', decision_function_shape = 'ovr') # 80%

		self.model = clf.fit(self.train_X, self.train_Y)
		print("svm classifier model type: " + str(type(self.model)))
		print(self.model)
		return

	def predict_test(self):
		print("predict test values")
		self.test_X = preprocessing.normalize(self.test_X)
		pred_Y = self.model.predict(self.test_X)
		print("predicted self.test_Y")

		correct = 0
		for i in range(len(pred_Y)):

			if pred_Y[i] == self.test_Y[i]:
				correct += 1

		accuracy = (correct / len(pred_Y)) * 100
		print("Accuracy : " + str(accuracy))
		print("CONFUSION MATRIX")

		con_matrix  = confusion_matrix(pred_Y, self.test_Y)
		print("Mat shape: " + str(con_matrix.shape))
		inp = np.array(con_matrix)
		ax=sns.heatmap(inp,annot=False,edgecolors="black",mask=inp==1,vmin=0,vmax=0.02)
		matplotlib.pyplot.show(ax)

		f = open("confusion_mat_hnd.txt", 'w')
		l = con_matrix.tolist()

		for v in l:
			s = str(v) + '\n'
			f.write(s)
		f.close()
		return

if __name__ == '__main__':
	co = HND_SVM('hndall.txt')
	co.read_all()
	co.split_data()
	
	co.train_classifier()
	co.predict_test()
	print("END OF PROGRAM")
