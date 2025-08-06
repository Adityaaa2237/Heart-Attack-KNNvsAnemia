import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

base_dir = 'C:/Users/ghaviranesya/Documents/Adit/data uas/Pemro/semester 2/Heart-Attack-Data-Set.xlsx'
data = pd.read_excel(base_dir)
data.head()
print(data)

# X dan Y
x = data.drop('target', axis = 1)
y = data['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle=True)
print('Jumlah Data Train : {0}'.format(len(x_train)))
print('Jumlah Data Test : {0}'.format(len(x_test)))

Neighbors = KNeighborsClassifier(n_neighbors= 3)
Neighbors.fit(x_train, y_train)

predictNeighbors = Neighbors.predict(x_test)
CM = confusion_matrix(y_test, predictNeighbors)
displayCM = ConfusionMatrixDisplay(confusion_matrix=CM)
displayCM.plot()
plt.show()

akurasi = metrics.accuracy_score(y_test, predictNeighbors)
sensivitas_recall = metrics.recall_score(y_test, predictNeighbors, pos_label = 1)
spesifikasi_recall = metrics.recall_score(y_test, predictNeighbors, pos_label = 0)
print('Akurasi :' +str(akurasi))
print('Sensivitas :' +str(sensivitas_recall))
print('Spesifikasi :' +str(spesifikasi_recall))
 
# true_positives = np.diag(CM)
# false_positives = np.sum(CM, axis=0) - true_positives
# false_negatives = np.sum(CM, axis=1) - true_positives

# precision = np.nan_to_num(np.divide(true_positives, (true_positives + false_positives)))
# recall = np.nan_to_num(np.divide(true_positives, (true_positives + false_negatives)))
# print(precision)
# print(recall)