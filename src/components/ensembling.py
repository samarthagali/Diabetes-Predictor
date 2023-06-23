import os
from glob import glob
import pickle
from dataclasses import dataclass
from lightgbm import LGBMClassifier


@dataclass
class ModelPath:
        models_path = "./models/"
        log_reg_path = "LogisticRegression.pkl"
        naive_bayes_path = "NaiveBayes.pkl"
        random_forest_path = "randomForest.pkl"
        svc_path = "SVC.pkl"
        lgbm_path = "lightGBM.pkl"

class Ensemble:
    def __init__(self, data):
        self.path = ModelPath()
        self.ensemble = []
        self.data = data
        print(os.getcwd())

    def logisticRegression(self):
         log_reg = pickle.load(open(self.path.models_path + self.path.log_reg_path, 'rb'))
         pred = log_reg.predict(self.data)
         pred = [0 if pred<0.5 else 1]
         self.ensemble.extend(pred)


    def naiveBayes(self):
         naive_bayes = pickle.load(open(self.path.models_path + self.path.naive_bayes_path, 'rb'))
         pred = naive_bayes.predict(self.data)
         self.ensemble.extend(pred)
    
    def randomForest(self):
         random_forest = pickle.load(open(self.path.models_path + self.path.random_forest_path, 'rb'))
         pred = random_forest.predict(self.data)
         self.ensemble.extend(pred)
        
    def SVC(self):
         svc = pickle.load(open(self.path.models_path + self.path.svc_path, 'rb'))
         pred = svc.predict(self.data)
         self.ensemble.extend(pred)
    
    def LGBM(self):
         lgbm = pickle.load(open(self.path.models_path + self.path.lgbm_path, 'rb'))
         pred = lgbm.predict(self.data)
         self.ensemble.extend(pred)
    
    def bagging(self):
         pred = sum(self.ensemble)/len(self.ensemble)
         return 0 if pred<0.5 else 1
    