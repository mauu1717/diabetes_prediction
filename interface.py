from flask import Flask,render_template,jsonify,request
import configure
from utils import diabetes
import traceback
import warnings
warnings.filterwarnings('ignore')
app = Flask(__name__)
@app.route('/')

def home():
    return render_template("index.html")


@app.route('/predict_diabetes',methods = ['GET','POST'])
def predict_diabetes():

    try:
        if request.method == 'POST':
            data = request.form.get
            
    
            print("user data is :", data)
            Glucose = eval(data('Glucose'))
            BloodPressure = eval(data('BloodPressure'))
            SkinThickness = eval(data('SkinThickness'))
            Insulin = eval(data('Insulin'))
            BMI = eval(data('BMI'))
            DiabetesPedigreeFunction = eval(data('DiabetesPedigreeFunction'))
            Age = eval(data('Age'))
            

    
            diabetes_pred = diabetes(Glucose, BloodPressure, SkinThickness, Insulin, BMI,DiabetesPedigreeFunction, Age)
            outcome = diabetes_pred.diabetes_prediction()

            # return jsonify({"Result" : f"outcome of diabetes test will be:{outcome}"})
            return  render_template('index.html',prediction = outcome)

        else:
            data = request.args.get
            print("User Data is ::::",data)
          
            Glucose = eval(data('Glucose'))
            BloodPressure = eval(data('BloodPressure'))
            SkinThickness = eval(data('SkinThickness'))
            Insulin = eval(data('Insulin'))
            BMI = eval(data('BMI'))
            DiabetesPedigreeFunction = eval(data('DiabetesPedigreeFunction'))
            Age = eval(data('Age'))
            
    
            diabetes_pred = diabetes(Glucose, BloodPressure, SkinThickness, Insulin, BMI,DiabetesPedigreeFunction, Age)
            outcome = diabetes_pred.diabetes_prediction()

            # return jsonify({"Result" : f"outcome of diabetes test will be:{outcome}"})
            return  render_template('index.html',prediction = outcome)
    
    except:
        print(traceback.print_exc())
        return  jsonify({"Message" : "Unsuccessful"})
    

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5003,debug=False)