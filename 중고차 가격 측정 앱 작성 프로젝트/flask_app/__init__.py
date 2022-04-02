from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

model = None

with open('model1.pkl','rb') as pickle_file:
   model = pickle.load(pickle_file)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/price', methods=['POST'])
def home():
    data1 = request.form['Kilometers_Driven']
    data2 = request.form['Engine']
    data3 = request.form['Power']
    data4 = request.form['Year']


    vector = np.vectorize(np.float)
    arr = np.array([[data1, data2,data3,data4]])
    arr = vector(arr)
    pred = model.predict(arr)
    pred = np.round(pred,4)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run()