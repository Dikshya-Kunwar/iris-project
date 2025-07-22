import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df=pd.read_csv('./data/Iris.csv')

df=df.drop(columns=['Id'])
df['Species']=df['Species'].str.replace('Iris-','',regex=False)

X= df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=df['Species']

X_train, X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=32)

model= RandomForestClassifier(n_estimators=17,random_state=32)
model.fit(X_train, y_train)

joblib.dump(model, 'model/iris_rf_model.pkl')

print('Model trained and saved as iris_rf.model.pkfl')

