import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("outbreak_detect.csv")

df = df.dropna()

LE = preprocessing.LabelEncoder()
# fitting it to our dataset
df.Outbreak = LE.fit_transform(df.Outbreak)


df = df.drop(['minTemp', 'maxTemp'], axis=1)


X = np.array(df[['avgHumidity', 'Rainfall', 'Positive', 'pf']])
Y = np.array(df[['Outbreak']])


sc = StandardScaler()
X = sc.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)


model = RandomForestClassifier()
model.fit(X_train, Y_train)


y_pred = model.predict(X_test)
print(y_pred)


pickle.dump(model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print('success')
