import pandas as pd
 
# membaca dataset dan mengubahnya menjadi dataframe
data = pd.read_csv('E:\KODINGAN\AKADEMI\DICODING\Machine Learning\sample_data\Salary_Data.csv')
print(data.info())
print('\n')
print(data.head())
print('\n')

import numpy as np
 
# memisahkan atribut dan label
X = data['YearsExperience']
y = data['Salary']
 
# mengubah bentuk atribut
X = np.array(X)
X = X[:,np.newaxis]

from sklearn.svm import SVR
 
# membangun model dengan parameter C, gamma, dan kernel
model  = SVR(C=1000, gamma=0.05, kernel='rbf')
 
# melatih model dengan fungsi fit
model.fit(X,y)

import matplotlib.pyplot as plt
 
# memvisualisasikan model
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.show()