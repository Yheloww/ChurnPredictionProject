from flask import Flask, request, render_template
import os 
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def main():
   return render_template("Home.html")

@app.route('/Home')
def home():
   return render_template("Home.html")

@app.route('/Predict', methods = ['POST', 'GET'])
def prediction():

   datas = dict()
   if (request.method == 'POST'):
      datas.update(request.form.to_dict())

      with open(os.path.join(os.path.abspath('models'), 'Churn_class.pkl'), 'rb') as f:
            model = pickle.load(f)

      
      
      proba = model.predict_proba(pd.DataFrame.from_dict([datas])).tolist()
      proba = proba[0][0]
      datas = f"The customer has a {round(proba*100)} % chance of churning"
      '''datas = model.predict(pd.DataFrame.from_dict([datas]))
      if(datas==1):
         datas = np.append(datas, "Churn")
      else:
         datas = np.append(datas, "Not churn")'''

      return render_template("Predict.html", result=datas)
   
   return render_template("Predict.html")

@app.route('/Dashboard')
def dashboard():
   return render_template("Dashboard.html")


if __name__ == '__main__':
   app.run()