from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from warnings import filterwarnings

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)

filterwarnings('ignore')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            pregnancies=float(request.form.get('pregnancies')),
            glucose=float(request.form.get('glucose')),
            bloodPressure=float(request.form.get('bloodPressure')),
            skinThickness=float(request.form.get('skinThickness')),
            insulin=float(request.form.get('insulin')),
            BMI=float(request.form.get('BMI')),
            diabetesPedigreeFunction=float(request.form.get('diabetesPedigreeFunction')),
            age=float(request.form.get('age'))
        )
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        result = 'Non-Diabetic' if not results else 'Diabetic'
        print("after Prediction")
        return render_template('home.html',results=result)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True, port='5000')        

