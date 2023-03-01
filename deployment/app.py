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

      with open('./models/test_cluster_class.pkl', 'rb') as f:
            model = pickle.load(f)

      prediction = model.predict(pd.DataFrame.from_dict([datas]))
      pred = prediction[0]
      if pred == 0: 
        pourcent = 6.0
      elif pred == 1:
        pourcent = 8.7
      else :
        pourcent = 33.8
      datas = f"The customer is from the cluster {pred}, and then has {pourcent}% chance of churning"


      '''proba = model.predict_proba(pd.DataFrame.from_dict([datas])).tolist()
      proba = proba[0][0]
      datas = f"The customer has a {round(proba*100)} % chance of churning"
      datas = model.predict(pd.DataFrame.from_dict([datas]))
      if(datas==1):
         datas = np.append(datas, "Churn")
      else:
         datas = np.append(datas, "Not churn")'''

      return render_template("Predict.html", result=datas)
   
   return render_template("Predict.html")

#The route of the dashboard page for the API. Show a use dashboard to see the state of all the datas
@app.route('/Dashboard')
def dashboard():
   return render_template("Dashboard.html")


if __name__ == '__main__':
   app.run()