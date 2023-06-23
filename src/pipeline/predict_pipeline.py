import sys
import os
import pandas as pd
from src.exceptions import CustomException
from src.util import load_object
from src.components.ensembling import Ensemble


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model = Ensemble(features)
            model.logisticRegression()
            model.naiveBayes()
            model.randomForest()
            model.SVC()
            model.LGBM()
            pred = model.bagging()
            return pred
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self, pregnancies, glucose, bloodPressure, skinThickness, 
                 insulin, BMI, diabetesPedigreeFunction, age):
        
        self.pregnancies = pregnancies
        self.glucose = glucose
        self.bloodPressure = bloodPressure
        self.skinThickness = skinThickness
        self.insulin = insulin
        self.BMI = BMI
        self.diabetesPedigreeFunction = diabetesPedigreeFunction
        self.age = age

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "pregnancies": [self.pregnancies],
                "glucose": [self.glucose],
                "bloodPressure": [self.bloodPressure],
                "skinThickness": [self.skinThickness],
                "insulin": [self.insulin],
                "BMI": [self.BMI],
                "diabetesPedigreeFunction": [self.diabetesPedigreeFunction],
                "age":[self.age]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)