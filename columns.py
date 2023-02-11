# coding: utf-8
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import configure
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import configure
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import configure
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import configure
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('configure.CSV_FILE_PATH')
df
df = pd.read_csv('configure.CSV_FILE_PATH')
df
df = pd.read_csv(configure.CSV_FILE_PATH)
df
df.info()
P = 0.8
np.log(P/(1-P))
P = 0.99
np.log(P/(1-P))
P = 0.1
np.log(P/(1-P))
P = 0.123456789
np.log(P/(1-P))
P = 0.0001
np.log(P/(1-P))
from statsmodels.stats.outliers_influence import variance_inflation_factor
variance_inflation_factor(df.to_numpy(), 3)
x = df.drop('Outcome', axis = 1)
y = df.Outcome

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, 
                                    random_state=55, stratify=y)
y.value_counts()
268/768
y_train.value_counts()
y_test.value_counts()
54/154
214/614
268/(768)
500/(768)
log_reg_model = LogisticRegression()
log_reg_model.fit(x_train, y_train) # Gredient Descent, LogLoss, With threshold of 0.5
log_reg_model.score(x_train, y_train)
log_reg_model.score(x_test, y_test)
plot_confusion_matrix(log_reg_model , x_train, y_train)
y_train.value_counts()
TP = 130
FN = 84
TN = 360
FP = 40
Recall = TP/(TP+FN)
Recall
Precision = TP/(TP+FP)
Precision
f1score = (2 * Precision * Recall)/ (Precision + Recall)
f1score
plt.figure(figsize=(5,3))
plot_confusion_matrix(log_reg_model , x_test, y_test) # Without prediction function
plt.savefig('Cnf_matrix_test.png')
## Evaluation on Training Dataset

y_pred_train = log_reg_model.predict(x_train)

cnf_matrix = confusion_matrix(y_train, y_pred_train)
print("Confusion Matrix :\n", cnf_matrix)
print("*"* 50)

accuracy = accuracy_score(y_train, y_pred_train)
print("Accuracy is :",accuracy)
print("*"* 50)

clf_report = classification_report(y_train, y_pred_train)
print('Classification Report :\n',clf_report)
print("*"* 50)

precision_value = precision_score(y_train, y_pred_train)
print("Precision :",precision_value)

recall_value = recall_score(y_train, y_pred_train)
print('Recall :',recall_value)

f1_value = f1_score(y_train, y_pred_train)
print('F1 Score:',f1_value)
print("*"* 50)
accuracy = (130 + 360) / 614
accuracy
sns.heatmap(cnf_matrix, annot=True)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix for Training Data')
plt.savefig('confusion_matrix_train.png')
cnf_matrix = confusion_matrix(y_train, y_pred_train)
print("Confusion Matrix :\n", cnf_matrix)
print("*"* 30)
TP = 130
FN = 84
TN = 360
FP = 40
P_class1 = TP/(TP+FP)
P_class1
P_class0 = TN/(TN+FN)
P_class0
R_class1 = TP/(TP+FN)
R_class1
R_class0 = TN/(TN + FP)
R_class0
f1_score_class1 = (2 * P_class1 * R_class1)/(P_class1 + R_class1)
f1_score_class1
f1_score_class0 = (2 * P_class0 * R_class0)/(P_class0 + R_class0)
f1_score_class0
clf_report = classification_report(y_train, y_pred_train)
print(clf_report)
macro_avg_precision = (0.81 + 0.76)/2
macro_avg_precision
macro_avg_recall = (0.90 + 0.61)/2
macro_avg_recall
macro_avg_f1 = (0.85 + 0.68)/2
macro_avg_f1
weighted_avg_precision = (0.81 * (400/614) + 0.76 * (214/614))
weighted_avg_precision
weighted_avg_recall = (0.90 * (400/614) + 0.61 * (214/614))
weighted_avg_recall
weighted_avg_f1 = (0.85 * (400/614) + 0.68 * (214/614))
weighted_avg_f1
# Evaluation on Testing Dataset

y_pred = log_reg_model.predict(x_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :\n", cnf_matrix)
print("*"* 50)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy is :",accuracy)
print("*"* 50)

clf_report = classification_report(y_test, y_pred)
print('Classification Report :\n',clf_report)
print("*"* 50)

precision_value = precision_score(y_test, y_pred)
print("Precision :",precision_value)

recall_value = recall_score(y_test, y_pred)
print('Recall :',recall_value)

f1_value = f1_score(y_test, y_pred)
print('F1 Score:',f1_value)
print("*"* 50)
y_pred_train_prob = log_reg_model.predict_proba(x_train)
y_pred_train_prob[30:35]
y_pred_train[30:35]
y_pred_train_prob[30:35][:,1]
x_train.shape
y_pred_train = log_reg_model.predict(x_train)
y_pred_train_prob = log_reg_model.predict_proba(x_train)
y_pred_train_prob.shape # 614,2
y_pred_train_prob_class1 = y_pred_train_prob[:,1]
fpr, tpr, threshold = roc_curve(y_train, y_pred_train_prob_class1)
# np.around(threshold,3)
# np.around(tpr,3)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
plt.savefig('ROC_Curve.png')
roc_auc_score(y_train, y_pred_train_prob_class1)
index = np.where(tpr >=0.90)[0][0]
index
fpr[index]
threshold[index]
def get_pred_class(threshold, pred_prob):
    print("Pred Prob is :",np.around(pred_prob,3))
    if pred_prob >= threshold:
        return "Class1"
    
    else:
        return "Class0"

threshold = 0.19
pred_prob = y_pred_train_prob_class1[54]

pred_class = get_pred_class(threshold, pred_prob)
print("Predicted class is :",pred_class)
px.scatter(x = fpr, y = tpr)
Glucose = 96.000
BloodPressure = 56.000
SkinThickness = 34.000
Insulin = 115.000
BMI = 24.700
DiabetesPedigreeFunction = 0.944
Age = 50.000
# Outcome = ?


test_array = np.array([Glucose,BloodPressure, SkinThickness, Insulin, BMI, 
                       DiabetesPedigreeFunction, Age], ndmin = 2)
test_array
pred_class = log_reg_model.predict(test_array)[0]
print("Predicted CLass using 0.5 Threshold is :",pred_class)
pred_class = log_reg_model.predict(test_array)
pred_class
fpr, tpr, threshold = roc_curve(y_train, y_pred_train_prob_class1)

def get_pred_class(thresh, pred_prob):
    print("Pred Prob is :",np.around(pred_prob,3))
    if pred_prob >= thresh:
        return "Class1"
    
    else:
        return "Class0"
  
index = np.where(tpr >=0.90)[0][0]

thresh = 0.19
thresh = threshold[index]
print("Best Threshold value is :",thresh)

pred_prob = log_reg_model.predict_proba(test_array)[:,1][0]

pred_class = get_pred_class(thresh, pred_prob)
print(f"Predicted class using {thresh} is : {pred_class}")
index
fpr, tpr, threshold = roc_curve(y_train, y_pred_train_prob_class1)
fpr
tpr
threshold.shape
log_reg_model.coef_
log_reg_model.intercept_
new_df = x_train.copy()
new_df["Log_Odds"] = np.log(y_pred_train_prob_class1/ (1-y_pred_train_prob_class1))
new_df.corr()
plt.figure(figsize=(20,5))
sns.heatmap(new_df.corr(), annot =True)
y_pred_train_prob_class1.shape
new_df["Log_Odds"]
with open(r"artifacts/regression_model.pkl",'wb') as f:
    pickle.dump(log_reg_model,f)
with open(r"artifacts/regression_model.pkl",'wb') as f:
    pickle.dump(log_reg_model,f)
with open(r"artifact/regression_model.pkl",'wb') as f:
    pickle.dump(log_reg_model,f)
import pickle
with open(r"artifact/regression_model.pkl",'wb') as f:
    pickle.dump(log_reg_model,f)
log_reg_model.n_features_in_
