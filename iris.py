import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib

df=pd.read_csv('./data/Iris.csv')

df=df.drop(columns=['Id'])
df['Species']=df['Species'].str.replace('Iris-','',regex=False)

X= df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=df['Species']

X_train, X_test,y_train,y_test= train_test_split(X,y,test_size=0.3)

model= RandomForestClassifier(n_estimators=15,random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model/iris_rf_model.pkl')

print('Model trained and saved as iris_rf.model.pkfl')

def predict(features):
    return model.predict(features)