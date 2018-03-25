import sys
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
from math import sqrt

DEBUG = False
TEST  = False

class KnnEntity():
	def __init__(self):
		"""self.data and self.label are pd.Series 
		with headers from raw data as index. 
		dtype: object"""
		self.data   = pd.Series()
		self.label  = pd.Series()

	def __str__(self):
		s = "@Label: "
		for i in self.label:
			s = s + str(i) + " "
		s = s + "||@Data: "
		for i in self.data.values:
			s = s + str(i) + " "
		return s

	def __eq__(self, other):
		temp1, temp2 = pd.Series(self.label.values), pd.Series(other.label.values)
		return temp1.equals(temp2)

class KnnTraining():
	'An KNN training module.'
	def __init__(self):
		self.source_file   = ""
		self.raw_data      = pd.DataFrame() # Need to import from files.
		self.headers       = list()         # Attributes/dimensions of KnnEntity.
		self.data_set      = pd.Series()
		self.__data_train  = pd.Series()
		self.__data_test   = pd.Series()
		self.__prediction  = pd.DataFrame(data = None, columns = ["Enti", "Label", "Pred"])

	# Public methods:
	def importCsv(self, filename):
		# Load in raw data from .csv file.
		self.source_file = filename
		try:
			self.raw_data = pd.read_csv(self.source_file)
		except:
			sys.stderr.write("Cannot import from %s.\n" % self.source_file)
		if DEBUG:
			sys.stderr.write("%s.source_file: %s.\n" % (KnnTraining.__name__, filename))
			sys.stderr.write("%s.raw_data is updated.\n" % KnnTraining.__name__)

	def loadDataSet(self, label_names):
		# Process data and store as a pd.Series of KnnEntity().
		labels = self.__extractLabel(label_names)
		self.__normalize()
		self.headers = self.raw_data.columns.values.tolist()
		self.raw_data.apply(self.__pushToEntity, axis = 1)
		self.data_set = self.data_set.drop([0])
		if DEBUG:
			sys.stderr.write("Removing NULL head...\n")
		for i in range(0, self.data_set.size):
			self.data_set.loc[i + 1].label = labels.loc[i]
		if DEBUG:
			sys.stderr.write("%s.data_set: %d entries added in total.\n\n" % (KnnTraining.__name__, self.data_set.size))
			print(self.data_set.head(10), file = sys.stderr)

	def shuffleData(self):
		self.data_set = shuffle(self.data_set)
		if DEBUG:
			sys.stderr.write("Random shuffle data set...\n")

	def setTrainingSet(self, train, test):
		# Parameters @train and @test are lists.
		self.__data_train = self.data_set.iloc[train]
		self.__data_test  = self.data_set.iloc[test]
		if DEBUG:
			sys.stderr.write("%s.data_train/.data_test are set.\n" % KnnTraining.__name__)

	def getAccuracy(self, k, add_weights = False, threshold = False, eliminate_tie = False):
		# @k: take k nearest neighbours.
		if DEBUG:
			sys.stderr.write("Training test data......\n")
		correct = 0
		total   = 0
		for i in self.__data_test:
			highest_votes = self.__getVotes(k, self.__getDistanceSet(i), add_weights, threshold, eliminate_tie)
			self.__prediction = self.__prediction.append(pd.DataFrame(data = {"Enti":  [i], 
																			  "Label": [i.label.values],
																			  "Pred":  [highest_votes]
																			  }), ignore_index = True)
			if (np.array_equal(i.label.values, highest_votes)):
				correct += 1
			total += 1
			if TEST:
				break
		accuracy = float(correct / total)
		sys.stderr.write("%s.__predition:\n" % KnnTraining.__name__)
		print(self.__prediction.head(25), file = sys.stderr)
		sys.stderr.write("Training accuracy: %f\n" % accuracy)
		return accuracy

	def getEntityNo(self):
		sys.stdout.write("This data set has %d entities in total.\n" % self.data_set.size)
		return self.data_set.size

	# Private methods:
	def __extractLabel(self, label_names):
		# Extract label after import raw data set.
		labels = self.raw_data.loc[:, label_names]
		self.raw_data = self.raw_data.drop(columns = label_names)
		if DEBUG:
			sys.stderr.write("%s.raw_data: Label extracted.\n" % KnnTraining.__name__)
		return labels

	def __normalize(self):
		self.raw_data = self.raw_data.apply(self.__normalizeInColumn, axis = 0)
		if DEBUG:
			# print(self.raw_data.head(), file = sys.stderr)
			sys.stderr.write("%s.raw_data: Successfully normalized working set.\n" % KnnTraining.__name__)

	def __getDistanceSet(self, testcase):
		distance = pd.DataFrame({"Enti":[], "Dist":[]})
		for i in self.__data_train:
			temp = pd.DataFrame({"Enti":[i], "Dist":[self.__distance(i, testcase)]})		
			distance = distance.append(temp, ignore_index = True)
		distance = distance.sort_values(by = ["Dist"], axis = 0, ascending = True)
		if DEBUG:
			sys.stderr.write("@distance set summary:\n")
			print(distance.describe(), file = sys.stderr)
		return distance

	def __getVotes(self, k, distance, add_weights = False, threshold = False, eliminate_tie = False):
		i = 0
		neighbours = list()
		n_labels   = list()
		n_dist     = list()
		while (i < k):
			neighbours.append(distance.iloc[i, distance.columns.get_loc("Enti")])
			drop = self.__threshold(distance.iloc[i, ], i, neighbours, threshold)
			self.__add_weights(n_dist, distance.iloc[i, ], drop, add_weights)
			n_labels.append(neighbours[i].label.values[0]) if drop == False else None
			k = (k + 1) if self.__eliminate_tie(k - i, drop, n_labels, eliminate_tie) else k
			i += 1
		votes = pd.DataFrame(data = {"Label": n_labels, "Dist": n_dist})
		votes_summary = pd.Series(data = votes.groupby(by = ["Label"])["Dist"].sum())
		highest_votes = list([votes_summary.idxmax()])
		if DEBUG:
			sys.stderr.write("@votes summary:\n")
			print(votes_summary, file = sys.stderr)
			print("Trained label: ", highest_votes, file = sys.stderr)
		return highest_votes

	def __pushToEntity(self, row):
		# Used in apply(). Convert data in DataFrame row to KnnEntity.
		entity = KnnEntity()
		entity.data = pd.Series(data = row.values, index = self.headers)
		self.data_set = self.data_set.append(pd.Series(entity), ignore_index = True)
		return self.data_set

	@staticmethod
	def __add_weights(n_dist, temp, drop, add_weights):
		if drop:
			return
		if (add_weights != False):
			n_dist.append(float(1 / (1 + temp.loc["Dist"])))
		else:
			n_dist.append(1)
		return

	@staticmethod
	def __threshold(temp, i, neighbours, threshold):
		if (threshold != False):
			if (temp.loc["Dist"] > threshold):
				if (i == 0):
					return False
				neighbours.pop()
				return True
			else:
				return False
		else:
			return False

	@staticmethod
	def __eliminate_tie(tail, drop, n_labels, eliminate_tie):
		if (eliminate_tie != False):
			if ((tail == 1) and (drop == False)):
				check_tie = pd.Series(data = n_labels).value_counts().duplicated(keep = False)
				try:
					tie = check_tie[True]
				except KeyError: 
					tie = False
				except IndexError:
					tie = False
				if DEBUG:
					print("Extended k when encounted a tie.\n", file = sys.stderr) if tie else None
				return True if tie else False
			else:
				return False
		else:
			return False

	@staticmethod
	def __normalizeInColumn(column):
		# Used in apply().
		column = column.astype(float, errors = "ignore")
		minimum_in_column = column.min()
		maximum_in_column = column.max()
		for i in range(0, column.size, 1):
			column.loc[i] = (column.loc[i] - minimum_in_column) / (maximum_in_column - minimum_in_column)
		return column

	@staticmethod
	def __distance(entity, testcase):
		sum = 0
		for i in range(0, testcase.data.size):
			sum = sum + pow((testcase.data.iloc[i] - entity.data.iloc[i]), 2)
		return sqrt(sum)


def main():
	wbcd = KnnTraining()
	wbcd.importCsv("wisc_bc_data.csv")

	data_drop = wbcd.raw_data.iloc[:, [0]]
	wbcd.raw_data = wbcd.raw_data.drop(columns = ["id"])

	wbcd.loadDataSet(label_names = ["diagnosis"])
	wbcd.getEntityNo()

	wbcd.shuffleData()
	wbcd.setTrainingSet(range(0,469), range(469,569))

	wbcd.getAccuracy(21, add_weights = True, eliminate_tie = True, threshold = 2)

if __name__ == "__main__":
	main()
