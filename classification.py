from sklearn import tree
feature = [[140,1], [130,1], [150,0], [170,0]]
label = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(feature, label)


#this function will predict that (160, 0) will be classified into group having label = 1
print clf.predict([[160, 0]])