import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus

#import dataset: Iris flower dataset
iris = load_iris()

#Setup testing data. Remove several examples as testing data.
# We'll use them to test how accurate the classifier is on data never seen before
test_idx = [0, 50, 100]
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#train classifier
irisClassifier = tree.DecisionTreeClassifier()
irisClassifier = irisClassifier.fit(train_data, train_target)
print ("===============================")
# Are test_target and predicted labels a match? 
print ("Actual labels for test target:", test_target)
#predict label for test_data
print ("Predicted labels on test_data:", irisClassifier.predict(test_data))
print ("===============================")

#visualize the tree
dot_data = StringIO()
tree.export_graphviz(irisClassifier, out_file=dot_data,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    impurity=False)


graph = pydotplus.pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

print(iris.feature_names, iris.target_names)

# test first
print (test_data[0], test_target[0])

#test second:
print (test_data[1], test_target[1])

#test third:
print (test_data[2], test_target[2])
