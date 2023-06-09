import numpy as np
import pandas as pd

x_test = pd.read_csv('C:\\Users\\USER\Desktop\\Python work space\\Data Analysis\\404_x_test.csv')
x_train = pd.read_csv('C:\\Users\\USER\Desktop\\Python work space\\Data Analysis\\404_x_train.csv')
y_train = pd.read_csv('C:\\Users\\USER\Desktop\\Python work space\\Data Analysis\\404_y_train.csv')

#데이터 전처리
print(x_train.info())
print(x_train.describe())

col_del = ['ID']
col_num = ['Age','Work_Experience','Family_Size']
col_cat = ['Gender','Ever_Married','Graduated','Profession','Spending_Score']
col_y = ['Segmentation']

from sklearn.model_selection import train_test_split 
x_tr, x_val, y_tr, y_val = train_test_split(x_train[col_num + col_cat], y_train[col_y].values.ravel(), 
                                            test_size = 0.3, stratify = y_train[col_y].values.ravel())

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(x_train[col_num])

x_tr[col_num] = scaler.transform(x_tr[col_num])
x_val[col_num] = scaler.transform(x_val[col_num])
x_test[col_num] = scaler.transform(x_test[col_num])

from sklearn.preprocessing import LabelEncoder 
x = pd.concat([x_train[col_cat], x_test[col_cat]])


for col in col_cat:
    le = LabelEncoder()
    le.fit(x[col])
    x_tr[col] = le.transform(x_tr[col])
    x_val[col] = le.transform(x_val[col])
    x_test[col] = le.transform(x_test[col])

    print(col)
    print(le.classes_)
    print('\n')


#모델 작성
## 1) RandomForest
from sklearn.ensemble import RandomForestClassifier 
modelRF = RandomForestClassifier(random_state=123)
modelRF.fit(x_tr, y_tr)

## 2) XGBoost
from xgboost import XGBClassifier 
modelXGB = XGBClassifier(random_state = 123)
modelXGB.fit(x_tr, y_tr)

#모델 평가
y_val_predRF = modelRF.predict(x_val)
y_val_predXGB = modelXGB.predict(x_val)

from sklearn.metrics import f1_score 
scoreRF = f1_score(y_val, y_val_predRF, average='macro')
scoreXGB = f1_score(y_val, y_val_predXGB, average='macro')

print('RandomForest: \t', scoreRF)
print('XGBoost: \t', scoreXGB)

def get_scores(model, x_tr, x_val, y_tr, y_val):
    y_tr_pred = model.predict(x_tr)
    y_val_pred = model.predict(x_val)
    tr_score = f1_score(y_tr, y_tr_pred, average='macro')
    val_score = f1_score(y_val, y_val_pred, average='macro')
    return f'train: {round(tr_score, 4)}, valid: {round(val_score, 4)}'

#제출
pred = modelXGB.predict(x_test[col_num + col_cat])
result = pd.DataFrame({'ID':x_test.ID, 'Segmentation':pred})
print(result)