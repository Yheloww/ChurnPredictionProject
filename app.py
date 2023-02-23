from flask import Flask, request, render_template
import os 
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
   return render_template("home.html")

@app.route('/Prediction', methods = ['POST', 'GET'])
def prediction():

   datas = dict()
   if (request.method == 'POST'):
      datas.update(request.form.to_dict())

      with open(os.path.join(os.path.abspath('models'), 'Tree_Class.pkl'), 'rb') as f:
            model = pickle.load(f)

      

      #datas = {'prediciton' : model.predict(pd.DataFrame.from_dict([datas])), 'status_code' : 0}

      return render_template("prediction.html", result=datas)
   
   return render_template("prediction.html")

@app.route('/Dashboard')
def dashboard():
   return render_template("dashboard.html")


if __name__ == '__main__':
   app.run()