import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = pd.read_excel('C:/Users/ghaviranesya/Documents/Adit/data uas/Pemro/semester 2/Heart-Attack-Data-Set.xlsx')
X = iris.drop(columns='target')
y = iris['target']
# from sklearn import datasets
# data,target = datasets.load_iris(return_X_y=True)

# from sklearn import datasets
# iris = pd.read_excel('C:/Users/ghaviranesya/Documents/Adit/data uas/Pemro/semester 2/Heart-Attack-Data-Set.xlsx')
# X = pd.DataFrame(iris.drop(columns='target'))
# X.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
# y= pd.DataFrame(iris.target)
# y.columns = ['Target']
# data,target = datasets.load_iris(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=484)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier,plot_tree
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

#model
plt.figure("Decision Tree", figsize=[12,5])
plot_tree(model, fontsize=10, filled=True)
# feature_names=iris.feature_names, class_names=iris.target_names
plt.tight_layout()
plt.show()

y_pred = model.predict(X_test)

labels_order = [0,1]
from sklearn.metrics import f1_score
f1 = f1_score(y_true=y_test, y_pred=y_pred, labels=labels_order, average="weighted")
print(f"f1: {f1}")

from sklearn.metrics import accuracy_score
from sklearn import metrics
akurasi = metrics.accuracy_score(y_test, y_pred)
sensivitas_recall = metrics.recall_score(y_test, y_pred, pos_label = 1)
spesifikasi_recall = metrics.recall_score(y_test, y_pred, pos_label = 0)
print(f"Accuracy: {akurasi}")
print(f"Sensivitas: {sensivitas_recall}")
print(f"Spesifikasi: {spesifikasi_recall}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
CM = confusion_matrix(y_test, y_pred)
displayCM = ConfusionMatrixDisplay(confusion_matrix=CM)
displayCM.plot()
plt.show()