# import our data
from mlxtend.data import loadlocal_mnist
import platform

if not platform.system() == 'Windows':
    X_train, y_train = loadlocal_mnist(
            images_path='C:\\Users\\Masha\\PycharmProjects\\pythonProject\\train-images.idx3-ubyte',
            labels_path='C:\\Users\\Masha\\PycharmProjects\\pythonProject\\train-images.idx3-ubyte')

else:
    X_train, y_train = loadlocal_mnist(
            images_path='C:\\Users\\Masha\\PycharmProjects\\pythonProject\\train-images.idx3-ubyte',
            labels_path='C:\\Users\\Masha\\PycharmProjects\\pythonProject\\train-labels.idx1-ubyte'
    )

if not platform.system() == 'Windows':
    x_test, y_test = loadlocal_mnist(
        images_path='C:\\Users\\Masha\\PycharmProjects\\pythonProject\\t10k-images.idx3-ubyte',
        labels_path='C:\\Users\\Masha\\PycharmProjects\\pythonProject\\t10k-labels.idx1-ubyte')

else:
    x_test, y_test= loadlocal_mnist(
        images_path='C:\\Users\\Masha\\PycharmProjects\\pythonProject\\t10k-images.idx3-ubyte',
        labels_path='C:\\Users\\Masha\\PycharmProjects\\pythonProject\\t10k-labels.idx1-ubyte'
    )
print('train data features:',X_train.shape, 'train data labels:',y_train.shape)
print('test data features:',x_test.shape, 'test data labels:',y_test.shape)
#X_train=X_train[0:10000,:]
#y_train=y_train[0:10000]
#x_test=x_test[0:1000,:]
#y_test=y_test[0:1000]
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
# look at our data
from random import randrange
index=randrange(10000)
plt.imshow(X_train[index].reshape(28,28), cmap='gray')
print('this is observation number:', index)
print('this image is:' +str(y_train[index]))
#plt.show()
# improve our features using scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
x_test = scaler.transform(x_test)

print(X_train, x_test)
from sklearn. metrics import classification_report,accuracy_score, precision_score, recall_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier(n_neighbors=10)
model_KNN.fit(X_train, y_train)
print (model_KNN, "\n")
LogisticPredict = model_KNN.predict(x_test)
print('Predict labels:',LogisticPredict[:10])
print('Actual labels:',y_test[:10])

print('classification report:',classification_report(y_test, LogisticPredict))
print("Overall Accuracy KNN improved:",accuracy_score(y_test, LogisticPredict))
print("Overall Precision KNN improved:",precision_score(y_test, LogisticPredict, average='macro'))
print("Overall Recall KNN improved:",recall_score(y_test, LogisticPredict, average='macro'))
mcm = confusion_matrix(y_test, LogisticPredict)
print(mcm)
sns.heatmap(mcm, annot=False)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Predictions made with KNN')
plt.show()


