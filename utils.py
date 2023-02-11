
import pickle
import json
import configure
import numpy as np
class diabetes():
    def __init__(self,Glucose, BloodPressure, SkinThickness, Insulin, BMI,
       DiabetesPedigreeFunction, Age):
        self.Glucose = Glucose
        self.BloodPressure = BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin =Insulin
        self.BMI=BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age = Age
    def __load_model(self):
        with open(r"artifact/regression_model.pkl",'rb') as f:
            self.model = pickle.load(f) 
            print('self.model:',self.model)


        with open(r'artifact/project_data.json','r')as f:
            self.project_data = json.load(f)
            print('project_data:',self.project_data)
    
    def diabetes_prediction(self): 
        self.__load_model()
        test_array = np.zeros(self.model.n_features_in_)
        test_array[0] = self.Glucose
        test_array[1] = self.BloodPressure
        test_array[2] = self.SkinThickness
        test_array[3] = self.Insulin
        test_array[4] = self.BMI
        test_array[4] = self.DiabetesPedigreeFunction
        test_array[4] = self.Age
        
        print("Test Array is :",test_array)
        predicted_diabetes = np.around(self.model.predict([test_array])[0],3)
        print("Predicted Charges :", predicted_diabetes)
        return  predicted_diabetes

