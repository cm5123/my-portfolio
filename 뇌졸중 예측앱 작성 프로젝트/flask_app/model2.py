import pandas as pd
from keras.models import load_model
from sklearn.linear_model import LinearRegression
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

target = 'stroke'
encoder = OrdinalEncoder()

df = pd.read_csv('data.csv')
df = df.drop(['id','ever_married','work_type','Residence_type','smoking_status'], axis = 1)
df = df.dropna(axis=0)

train, test = train_test_split(df, train_size=0.80, test_size=0.20, 
                              stratify=df[target], random_state=2)

train, val = train_test_split(df, train_size=0.80, test_size=0.20, 
                              stratify=df[target], random_state=2)

train.shape, val.shape, test.shape

features = train.drop(columns=[target]).columns

X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]
X_test = test[features]

enc = OneHotEncoder()
imp_mean = SimpleImputer()
scaler = StandardScaler()
model_lr = LogisticRegression(n_jobs=-1)

X_train_encoded = enc.fit_transform(X_train)
X_train_imputed = imp_mean.fit_transform(X_train_encoded)
X_train_scaled = scaler.fit_transform(X_train_imputed)
model_lr.fit(X_train_scaled, y_train)

X_val_encoded = enc.transform(X_val)
X_val_imputed = imp_mean.transform(X_val_encoded)
X_val_scaled = scaler.transform(X_val_imputed)


X_test_encoded = enc.transform(X_test)
X_test_imputed = imp_mean.transform(X_test_encoded)
X_test_scaled = scaler.transform(X_test_imputed)

y_pred = model_lr.predict(X_test_scaled)

pipe = make_pipeline(
    OrdinalEncoder(), 
    DecisionTreeClassifier(max_depth=5, random_state=2)
)

pipe.fit(X_train, y_train)
print('검증 정확도', pipe.score(X_val, y_val))
print('f1 score', f1_score(y_val, pipe.predict(X_val)))

custom = len(y_train)/(2*np.bincount(y_train))
custom

params = {
    'max_depth': [2, 3, 4],
    'min_samples_split': [2, 3, 4]
}

dtc = DecisionTreeClassifier()

grid_tree = GridSearchCV(dtc, param_grid=params, cv=3, refit=True)
grid_tree.fit(X_train_encoded , y_train)
print('best parameters : ', grid_tree.best_params_)
print('best score : ', grid_tree.best_score_)
em = grid_tree.best_estimator_
pred = em.predict(X_val_encoded)
accuracy_score(y_val, pred)

pipe = make_pipeline(
    OrdinalEncoder(), 
    DecisionTreeClassifier(max_depth=2, min_samples_split=2, class_weight={False:custom[0],True:custom[1]}, random_state=2)
)


pipe.fit(X_train, y_train)
print('검증 정확도: ', pipe.score(X_val, y_val))
print('f1 score', f1_score(y_val, pipe.predict(X_val)))

model = DecisionTreeClassifier(max_depth=2, min_samples_split=2, class_weight={False:custom[0],True:custom[1]}, random_state=2)

model.fit(X_train_encoded, y_train)

import pickle

with open('model.pkl','wb') as pickle_file:
    pickle.dump(model, pickle_file)