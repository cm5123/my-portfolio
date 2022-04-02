import pandas as pd
from keras.models import load_model
from sklearn.linear_model import LinearRegression
import numpy as np
import tensorflow as tf
import pickle

df = pd.read_csv('tfile.csv')

df[['Kilometers_Driven','Engine','Power','Price','Year']]
df = df.dropna(axis=0)

df['Engine'] = df['Engine'].replace(' CC','', regex=True)
df['Power'] = df['Power'].replace(' bhp','', regex=True)

model = LinearRegression()

feature = ['Kilometers_Driven','Engine','Power','Year']
target = ['Price']
X_train = df[feature]
y_train = df[target]

model.fit(X_train, y_train)

import pickle

with open('model.pkl','wb') as pickle_file:
    pickle.dump(model, pickle_file)
