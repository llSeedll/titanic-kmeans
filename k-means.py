import numpy as np
import pandas as pd
from sklearn import preprocessing
'''
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

X = np.array([
		[1, 2],
		[1.5, 1.8],
		[5, 8],
		[8, 8],
		[1, 0.6],
		[9, 11] ])
'''
'''
plt.scatter(X[:, 0], X[:, 1], s=100)
plt.show()
'''

#colors = 10*["g", "r", "c", "b", "k"]



class K_Means:

	def __init__(self, k=2, tol=1e-3, max_iter=1000):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self, data):
		self.centroids = {}
		for i in range(self.k):
			self.centroids[i] = data[i]
		for i in range(self.max_iter):
			self.classifications = {}
			for i in range(self.k):
				self.classifications[i] = []

			for featureset in data:
				distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)

			prev_centroids = dict(self.centroids)

			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)

			optimized = True

			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]
				if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
					optimized = False
					break

			if optimized:
				break


	def predict(self, data):
		distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
		return distances.index(min(distances))

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.fillna(0, inplace=True)

def handle_non_numeric_data(df):
	columns = df.columns.values
	for column in columns:
		text_digit_values = {}
		def convert_to_int(val):
			return text_digit_values[val]

		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_values:
					text_digit_values[unique] = x
					x += 1

			df[column] = list(map(convert_to_int, df[column]))

	return df

df = handle_non_numeric_data(df)
df.drop(['ticket', 'home.dest'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = K_Means()
clf.fit(X)

correct = 0
for i in range(len(X)):
	toPredict = np.array(X[i].astype(float))
	toPredict = toPredict.reshape(-1, len(toPredict))
	prediction = clf.predict(toPredict)
	if prediction == y[i]:
		correct += 1

accuracy = max(correct/len(X), 1 - (correct/len(X)))
print('Accuracy', accuracy)

'''
for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], s=110, marker="o", color="k", linewidths=5)

for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], s=110, marker="x", color=color, linewidths=5)

unknowns = np.array([
		[1, 3],
		[3, 3],
		[4, 2],
		[5, 9],
		[8, 9],
		[8, 8]
	])

for unknown in unknowns:
	classification = clf.predict(unknown)
	plt.scatter(unknown[0], unknown[1], s=110, marker="*", color=colors[classification], linewidths=5)

plt.show()
'''
