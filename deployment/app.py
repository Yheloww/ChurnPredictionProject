from flask import Flask, request, render_template
import os
import pickle
import pandas as pd
import numpy as np

#Build the API.
app = Flask(__name__)

#The route of the home page for the API.
@app.route('/')
def main():
   return render_template("Home.html")

#Idem.
@app.route('/Home')
def home():
   return render_template("Home.html")

#The roure of the predict page for the API. Take all the datas from the form 
#and transfer them to a prediction to see the % of chance for a client to churn.
@app.route('/Predict', methods = ['POST', 'GET'])
def prediction():

   datas = dict()
   if (request.method == 'POST'):
      datas.update(request.form.to_dict())

      with open('./models/Churn_cluster_3_final.pkl', 'rb') as f:
            model = pickle.load(f)

      prediction = model.predict(pd.DataFrame.from_dict([datas]))[0]
      dict_percentages = {0: 8.1, 1 : 5.3, 2 : 30.1}
      perc = dict_percentages[prediction]
      datas = f'the client is from cluster {prediction} then his probability of churning is {perc} %'

      return render_template("Predict.html", result=datas)
   
   return render_template("Predict.html")

#The route of the dashboard page for the API. Show a use dashboard to see the state of all the datas
@app.route('/Dashboard')
def dashboard():
   return render_template("Dashboard.html")


if __name__ == '__main__':
   app.run()