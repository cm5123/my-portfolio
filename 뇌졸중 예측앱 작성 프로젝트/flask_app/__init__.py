from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

model = None

with open('model.pkl','rb') as pickle_file:
   model = pickle.load(pickle_file)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/value', methods=['POST'])
def home():
    data1 = request.form['gender']
    data2 = request.form['age']
    data3 = request.form['hypertension']
    data4 = request.form['heart_disease']
    data5 = request.form['avg_glucose_level']
    data6 = request.form['bmi']


    vector = np.vectorize(np.float)
    arr = np.array([[data1,data2,data3,data4,data5,data6]])
    arr = vector(arr)
    pred = model.predict(arr)
    return render_template('after.html', data=pred)



if __name__ == "__main__":
    app.run()