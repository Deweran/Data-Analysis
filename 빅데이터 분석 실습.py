#필요한 패키지 가져오기
import seaborn as sns
import pandas as pd
import numpy as np

#data = sns.load_dataset('mpg') seaborn 패키지에서 제공하는 원본 데이터셋(여기서는 사용x)
#print(data.head())

#데이터 불러오기
x_test = pd.read_csv('C:/Users/USER/Desktop/Python work space/Data Analysis/mpg_X_test.csv')
x_train = pd.read_csv('C:/Users/USER/Desktop/Python work space/Data Analysis/mpg_X_train.csv')
y_train = pd.read_csv('C:/Users/USER/Desktop/Python work space/Data Analysis/mpg_y_train.csv')

#데이터 전처리
print(x_train.info())

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x_train[['horsepower']] = imputer.fit_transform(x_train[['horsepower']])
x_test[['horsepower']] = imputer.fit_transform(x_test[['horsepower']])

print(x_train.describe())

col_del = ['name']
col_num = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year']
col_cat = []
col_y = ['isUSA']

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

#데이터 모형 구축
#데이터 분할
from sklearn.model_selection import train_test_split

x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.3) 

#데이터 스케일링
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_tr[col_num])
x_tr[col_num] = scaler.transform(x_tr[col_num])
x_val[col_num] = scaler.transform(x_val[col_num])
x_test[col_num] = scaler.transform(x_test[col_num])

#KNN, 의사결정 나무 모형 설정
from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
modelKNN.fit(x_tr, y_tr.values.ravel())

from sklearn.tree import DecisionTreeClassifier
modelDT = DecisionTreeClassifier(max_depth=10)
modelDT.fit(x_tr, y_tr)

#데이터 모형 평가
y_val_pred = modelKNN.predict(x_val)

y_val_pred_probaKNN = modelKNN.predict_proba(x_val)
y_val_pred_probaDT = modelDT.predict_proba(x_val)

from sklearn.metrics import roc_auc_score

scoreKNN = roc_auc_score(y_val, y_val_pred_probaKNN[:,1])
scoreDT = roc_auc_score(y_val, y_val_pred_probaDT[:,1])

print(scoreKNN, scoreDT)

#사용자가 직접 하이퍼파라미터를 학습하는 모형 저장
best_model = None
best_score = 0

for i in range(2, 10):
    model = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
    model.fit(x_tr, y_tr.values.ravel())
    y_val_pred_proba = model.predict_proba(x_val)
    score = roc_auc_score(y_val, y_val_pred_proba[:,1])
    print(i," 개의 이웃 확인 : ", score)
    if best_score <= score:
        best_model = model

print(best_model.predict_proba(x_test))

pred = best_model.predict_proba(x_test)[:,1]
print(pred)

pd.DataFrame({'isUSA':pred}).to_csv('C:/Users/USER/Desktop/Python work space/Data Analysis/03000000.csv', index=False)