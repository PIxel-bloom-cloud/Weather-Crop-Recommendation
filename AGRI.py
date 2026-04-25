import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("crop.CSV")

X = data[['temperature','rainfall']]
y = data['crop']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt_pred = dt.predict(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)

print("DT Accuracy:", accuracy_score(y_test,dt_pred))
print("KNN Accuracy:", accuracy_score(y_test,knn_pred))