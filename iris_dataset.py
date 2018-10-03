# https://www.youtube.com/watch?v=YcUPVziBsMA&index=2&list=PL9ooVrP1hQOHUfd-g8GUpKI3hHOwM_9Dn

#check the version of libraries


# #python version
# import sys
# print('Python: {}'.format(sys.version))

# #scipy
# import scipy
# print('scipy: {}'.format(scipy.__version__))

# #numpy
# import numpy
# print('numpy: {}'.format(numpy.__version__))

# #matplotlib
# import matplotlib
# print('matplotlib: {}'.format(matplotlib.__version__))

# #pandas
# import pandas
# print('pandas: {}'.format(pandas.__version__))

# #scikit-learn
# import sklearn
# print('sklearn: {}'.format(sklearn.__version__))

# load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt 
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# #data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# print rows and column count of dataset
# print (dataset.shape)

#print 1st 30 rows of dataset
# print (dataset.head(30))

# print more descriptive values of dataset
# print (dataset.describe())

# print no. of instances in each class
# print (dataset.groupby('class').size())

# Graph ---> Uni-varient plot
# line-----
# dataset.plot(kind='line', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# box
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# histogram
dataset.hist()
plt.show()

# Graph ---> Multi-varient plot
# create scatter matrix
# scatter_matrix(dataset)
# plt.show()

# Lets analyse some algorithm and see accuracy of different algorithm in data analysis
# 1st create validation dataset(taining dataset) - 1st 80% of data will train data, next 20% will validate data
# array = dataset.values
# X = array[:, 0:4]
# Y = array[:, 4]
# validation_size = 0.20
# seed = 6
# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)

# seed = 6
# scoring = 'accuracy'


# #Check 5 different algorithm (Spot check algorithms)
# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))

# # evaluate each model in turn
# results = []
# names = []
# for name, model in models:
# 	kfold=model_selection.KFold(n_splits=10, random_state=seed)
# 	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
# 	results.append(cv_results)
# 	names.append(name)
# 	msg="%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# 	print(msg)









