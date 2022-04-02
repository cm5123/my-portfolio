import pandas as pd
from keras.models import load_model
from sklearn.linear_model import LinearRegression
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')

df = df.drop(['id','ever_married','work_type','Residence_type','smoking_status'], axis = 1)
df = df.dropna(axis=0)

model = LinearRegression()

target = ['stroke']
feature = ['age','hypertension','heart_disease','avg_glucose_level','bmi']
X_train = df[feature]
y_train = df[target]

model.fit(X_train, y_train)

import pickle

with open('model.pkl','wb') as pickle_file:
    pickle.dump(model, pickle_file)
