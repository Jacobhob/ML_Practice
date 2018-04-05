# @Author: Jacob Lu
# @Date: 3/27/2018
# @Usage: ./SVM.py auto {int}

import sys
import os
from PIL import Image
import numpy as np 
import pandas as pd 
import re
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from concurrent.futures import ThreadPoolExecutor

DEBUG     = False
MULTHREAD = True
SAMPLES   = 448
TEST      = 48
RESIZE = (128,128)
c      = 2.0    # 1
gamma  = 0.000030517578125    # 1/(RESIZE[0]*RESIZE[1])
samples_path   = "./signature/samples/"
svmtrain_exe   = "./libsvm/svm-train"
svmpredict_exe = "./libsvm/svm-predict"
grid_py        = "./libsvm/grid.py"

class sigNode:
	"""Nodes of signature samples."""
	count = 0

	def __init__(self, filename):
		sigNode.count += 1
		self.id    = sigNode.count
		self.label = True
		self.img   = self.__readImg(filename)
		self.__plotImg(self.img) if DEBUG else None

	def __readImg(self, filename):
		try:
			"""Read image, convert to gray scale and resize."""			
			img = Image.open(filename).convert('1').resize(RESIZE)
			self.label = True if re.compile(r"[T,F]+").search(filename).group(0) == 'T' else False
			sys.stderr.write("Successfully read #%d (%r).\n" % (self.id, self.label)) if DEBUG else None
			return img
		except:
			sys.stderr.write("Cannot read or convert %d.jpg.\n" % self.id)

	@staticmethod
	def __plotImg(img):
		print(np.array(img).shape)
		print(np.array(img).dtype)
		print(np.arrary(img).size)
		print(type(np.array(img)))
		plt.imshow(img)
		plt.axis('off')
		plt.show()


def inputDataset(normalize = True):
	df, df_normalize, df_m = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(data = None, index = [0,1], columns = range(0,RESIZE[0]*RESIZE[1]+1))

	def normalizeInColumn(column):
		# Used in apply().
		column = column.astype(int, errors = "ignore")
		minimum_in_column, maximum_in_column = column.min(), column.max()
		for i in range(0, column.size, 1):
			if minimum_in_column != maximum_in_column:
				column.loc[i] = 2*(column.loc[i]-minimum_in_column)/(maximum_in_column-minimum_in_column)-1
			elif maximum_in_column == 1:
				column.loc[i] = 1
			elif minimum_in_column == 0:
				column.loc[i] = -1
		return column

	def MTdo(row):
		# Used in Multithread Pool.
		for i in range(1, df.shape[1], 1):
			# minimum_in_column, maximum_in_column = df.iloc[:,i].min(), df.iloc[:,i].max()
			minimum_in_column, maximum_in_column = df_m.iloc[0, i], df_m.iloc[1, i]
			if minimum_in_column != maximum_in_column:
				row.iloc[i] = str(i) + ':' + str(2*(row.iloc[i]-minimum_in_column)/(maximum_in_column-minimum_in_column)-1)
			elif maximum_in_column == 1:
				row.iloc[i] = str(i) + ':' + str(1)
			elif minimum_in_column == 0:
				row.iloc[i] = str(i) + ':' + str(-1)
		return row

	filenames = os.popen("ls "+samples_path+"*.jpg")
	for filename in filenames:
		signature = sigNode(filename.strip())
		df = df.append(pd.Series(np.append(signature.label, np.array(signature.img).flatten())), ignore_index = True).astype(int)
	sys.stderr.write("Dataset successfully imported.\n")
	print(df.shape)	if DEBUG else None
	if normalize:
		# Normalize data to [-1,1]
		if not MULTHREAD:
			df = df.apply(normalizeInColumn, axis = 'index')
		else:
			for i in range(0, df.shape[1], 1):
				df_m.iloc[0, i], df_m.iloc[1, i] = df.iloc[:,i].min(), df.iloc[:, i].max()
			with ThreadPoolExecutor(max_workers = 4) as executor:
				result = executor.map(MTdo, pd.Series(df.iloc[i,:] for i in range(0, df.shape[0])))
			for i in result:
				df_normalize = df_normalize.append(i) 
			df = df_normalize
		sys.stderr.write("Dataset has been successfully normalized.\n")
	df.to_csv(samples_path+"all_samples", sep = ' ', index = False, header = False)


def randomShuffle(no_of_test = TEST):
	df = pd.read_csv(samples_path+"all_samples", header = None)
	df = shuffle(df)
	print(df.head()) if DEBUG else None
	sys.stderr.write("Successfully shuffled.\n")
	df_train = df.iloc[:-no_of_test]
	df_train.to_csv("./train", index = False, header = False)
	df_test  = df.tail(no_of_test)
	df_test.to_csv("./test",   index = False, header = False)


def gridSearch(dataset = "./train"):
	cmd = "{0} -svmtrain '{1}' -gnuplot  null '{2}'".format(grid_py, svmtrain_exe, dataset)
	if DEBUG:
		excecute = os.system(cmd)
	else:
		f = os.popen(cmd)
		line = ''
		while True:
			last_line = line
			line = f.readline()
			if not line: break
		global c, gamma
		c, gamma, rate = map(float, last_line.split())
		print("C =", c, "Gamma =", gamma, "Rate =", rate)


def SVMtrain(c = c, gamma = gamma, dataset = "./train"):
	cmd = "{0} -c '{1}' -g '{2}' '{3}'".format(svmtrain_exe, c, gamma, dataset)
	execute = os.system(cmd)


def predict(dataset = "./test"):
	cmd = "{0} '{1}' ./train.model ./test.out".format(svmpredict_exe, dataset)
	execute = os.system(cmd)


def test():
	return

if __name__ == "__main__":
	if sys.argv[1] == "test":
		test()
	elif sys.argv[1] == "import":
		inputDataset()
	elif sys.argv[1] == "shuffle":
		randomShuffle()
	elif sys.argv[1] == "gridsearch":
		gridSearch()
	elif sys.argv[1] == "train":
		SVMtrain()
	elif sys.argv[1] == "predict":
		predict()
	elif sys.argv[1] == "auto":
		if (len(sys.argv) >= 3):
			count = int(sys.argv[2])
		else:
			count = 1
		inputDataset()
		for i in range(0, count, 1):
			randomShuffle()
			gridSearch()
			SVMtrain()
			predict()
			# SVMtrain(c = 2, gamma = 0.00048828125)
			# predict()
			# SVMtrain(c = 2, gamma = 0.000030517578125)
			# predict()

