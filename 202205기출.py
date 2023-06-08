import pandas as pd
import numpy as np

x_test = pd.read_csv('C:\\Users\\USER\\Desktop\\Python work space\\Data Analysis\\504_x_test.csv')
x_train = pd.read_csv('C:\\Users\\USER\\Desktop\\Python work space\\Data Analysis\\504_x_train.csv')
y_train = pd.read_csv('C:\\Users\\USER\\Desktop\\Python work space\\Data Analysis\\504_y_train.csv')

#print(x_train.head(5))
print(x_train.info())
print(x_train.describe())
print(y_train.info())

#print(x_train.isnull().sum())

col_del = ['id']
col_num = ['year','mileage','tax','mpg','engineSize']
col_cat = ['model','transmission','fuelType']
col_y = ['price']

#데이터 분할
from sklearn.model_selection import train_test_split

x_tr, x_val, y_tr, y_val = train_test_split(x_train[col_num + col_cat],
                                             y_train[col_y].values.ravel(), test_size=0.3)
print(x_tr.head(5))

#수치형 변수 스케일링
from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()
scaler.fit(x_tr[col_num])

x_tr[col_num] = scaler.transform(x_tr[col_num])
x_val[col_num] = scaler.transform(x_val[col_num])
x_test[col_num] = scaler.transform(x_test[col_num])

#범주형 변수 인코딩
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

# 1) 랜덤 포레스트
from sklearn.ensemble import RandomForestRegressor
modelRF = RandomForestRegressor(random_state = 123)
modelRF.fit(x_tr, y_tr)

# 2) XGBoost
from xgboost import XGBRegressor
modelXGB = XGBRegressor(objective = 'reg:squarederror', random_state = 123)
modelXGB.fit(x_tr, y_tr)

#데이터 모형 평가
y_val_modelRF = modelRF.predict(x_val)
y_val_modelXGB = modelXGB.predict(x_val)

#답안 채점기준 검증 데이터 평가
from sklearn.metrics import mean_squared_error

def cal_rmse(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred))

scoreRF = cal_rmse(y_val, y_val_modelRF)
scoreXGB = cal_rmse(y_val, y_val_modelXGB)

print('Random Forest:  \t', scoreRF)
print('XGBoost:  /t', scoreXGB)


#결과 제출
pred = modelXGB.predict(x_test[col_num+col_cat])
result = pd.DataFrame({'id':x_test.id, 'price':pred})

print(result.head(5))

