import csv 
import random
import math
import operator
import pandas


def loadDataset(inputFile, split, trainingSet=[], testSet=[]):
	with open(inputFile) as ifile:
		lines = csv.reader(ifile) #reading csv file
		dataset = list(lines)

		for x in range(len(dataset) - 1):
			for y in range(4): #read 4 coloumns
				dataset[x][y] = float(dataset[x][y])

			if random.random() < split:
				trainingSet.append(dataset[x][y])
			else: 
				testSet.append(dataset[x][y])

# in order to calculate similarity b/w two data instances. For that we need distance of its k neighbour datapoints
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)

	return math.sqrt(distance)

# find k neighbours of testData in trainingData set
def getNeighbors(trainingDataset, testDataset, k):
	distances = {}
	testLength = len(testDataset) - 1
	for x in range (len(trainingDataset)):
		dist = euclideanDistance(testDataset, trainingDataset, testLength)
		distances.append(trainingDataset[x], dist)
		# distances[trainingDataset[x]] = dist

	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])

	return neighbors

# predict response based on neighbors
def getResponse(neighbors):
	classVotes = {} # this is dictionary. Kind of hashMap("neighbor_iris_name", count of that iris in neighborhood)
	for x in range(len(neighbors)):
		response = neighbors[x][-1]

		if(response in classVotes): 
			classVotes[response] += 1
		else: 
			classVotes[response] = 1

	sortedVotes = sorted(classVotes.items(), key = operator.itemgetter(1), reverse = True)
	return sortedVotes[0][0] #this will return classVote having highest no. of neighbor

# to predict how accurate is our algorithm. Accuracy = correctCases/totalCases
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1

	return (correct/float(len(testSet))) * 100.0



def main():
	# prepare data
	trainingSet = []
	testSet = []
	split = 0.67      # 2/3 training data, 1/3 test data

	loadDataset('iris_data.txt', split, trainingSet, testSet)

	print ('Train Set: '+ repr(len(trainingSet)))
	print ('Test Set: '+ repr(len(testSet)))

	#generate predictions
	predictions = []
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)

		print('> predicted = '+ repr(result) + ', actual = '+ repr(testSet[x][-1]))

	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy : '+repr(accuracy) + "%")


main()






















