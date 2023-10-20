import pandas as pd
import numpy as np

# 학생성적 예측 데이터
# 데이터 설명 : 학생성적 예측 (종속변수 :G3)
x_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/X_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/y_train.csv")
x_test= pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/X_test.csv")

print(x_train.head())
print(y_train.head())

# 풀이

#x_train.isnull().sum()
#x_test.isnull().sum()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error , mean_absolute_error , mean_absolute_percentage_error ,r2_score

drop_col = ['StudentID']

x_train_drop = x_train.drop(columns = drop_col)
x_test_drop = x_test.drop(columns = drop_col)
y_train_target = y_train['G3']

x_train_dum = pd.get_dummies(x_train_drop)
x_test_dum = pd.get_dummies(x_test_drop)[x_train_dum.columns]


xtr , xt , ytr, yt = train_test_split(x_train_dum , y_train_target)

rf = RandomForestRegressor(random_state =42)
rf.fit(xtr,ytr)

y_validation_pred = rf.predict(xt)

# 모델평가 
# mse : mean_squared_error / mae : mean_absolute_error  / mape : mean_absolute_percentage_error
# rmse : root_mean_squerd_error -> 패키지 없음 np.sqrt(mean_squared_error) 해줘야함

# y_true ,y_pred 순서 help로 잘 확인 하시고 사용하셔요

#mse 
print('validation mse' ,mean_squared_error(yt,y_validation_pred))

#mae 
print('validation mae' ,mean_absolute_error(yt,y_validation_pred))

#mape 
print('validation mape' ,mean_absolute_percentage_error(yt,y_validation_pred))

#rmse
print('validation rmse' ,np.sqrt(mean_absolute_percentage_error(yt,y_validation_pred)))

#r2
print('validation r2 score' ,r2_score(yt,y_validation_pred))
