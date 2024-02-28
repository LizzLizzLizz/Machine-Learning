#konversi data menggunakan pandas dataframe
import pandas as pd
df = pd.read_csv('E:\KODINGAN\AKADEMI\DICODING\Machine Learning\sample_data\california_housing_train.csv')
head = df.head()
print(head)
print('\n')

#Normalization data
from sklearn.preprocessing import MinMaxScaler
data = [[12000000, 33], [35000000, 45], [4000000, 23], [6500000, 26], [9000000, 29]]
scaler = MinMaxScaler()
scaler.fit(data)
print(scaler.transform(data))
print('\n')

#stadndarization data
from sklearn import preprocessing
data = [[12000000, 33], [35000000, 45], [4000000, 23], [6500000, 26], [9000000, 29]]
scaler = preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)
print(data)
print('\n')

#train test split
from sklearn import datasets

iris = datasets.load_iris()

x=iris.data
y=iris.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

print(len(x))
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))



