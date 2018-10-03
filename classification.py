from sklearn import tree
feature = [[140,1], [130,1], [150,0], [170,0]]
label = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(feature, label)

print clf.predict([[160, 0]])