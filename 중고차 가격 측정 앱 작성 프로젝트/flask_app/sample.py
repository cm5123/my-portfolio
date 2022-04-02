import pandas as pd
from keras.models import load_model
from sklearn.linear_model import LinearRegression
import numpy as np
import tensorflow as tf
import pickle

# 주어진 url 주소를 이용해 house prices 데이터를 가져옵니다.
df = pd.read_csv('https://ds-lecture-data.s3.ap-northeast-2.amazonaws.com/house-prices/house_prices_train.csv')
df_t = pd.read_csv('https://ds-lecture-data.s3.ap-northeast-2.amazonaws.com/house-prices/house_prices_test.csv')

## 예측모델 인스턴스를 만듭니다
model = LinearRegression()

## X 특성들의 테이블과, y 타겟 벡터를 만듭니다
feature = ['GrLivArea','OverallQual','OverallCond']
target = ['SalePrice']
X_train = df[feature]
y_train = df[target]

## 모델을 학습(fit)합니다
model.fit(X_train, y_train)

## 새로운 데이터 한 샘플을 선택해 학습한 모델을 통해 예측해 봅니다
# X_test = [[4000,500,30]]
# y_pred = model.predict(X_test)

# print(f'{X_test} 차의 예상 가격은 ${int(y_pred)} 입니다.')


import pickle

with open('model.pkl','wb') as pickle_file:
    pickle.dump(model, pickle_file)