from sklearn import datasets
from sklearn.model_selection import train_test_split
import sys

rate=float(sys.argv[1])
print("test size : {}%".format(int(100*rate)))

iris=datasets.load_iris()
print("data size : {}".format(iris['data'].shape))

X=iris.data[:,[2,3]]
y=iris.target

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=rate, random_state=0)


from sklearn.tree import DecisionTreeClassifier

iris_tree=DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
iris_tree.fit(X_train, y_train)
print("training dataset : {:.3f}".format(iris_tree.score(X_train, y_train)))
print("test dataset : {:.3f}".format(iris_tree.score(X_test, y_test)))