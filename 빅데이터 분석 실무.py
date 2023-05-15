import pandas as pd 

#1. 데이터 불러오기
data = pd.read_csv('C:/Users/USER/Desktop/Python work space/Data Analysis/housing_data.csv', header = None, sep=',')
col_names = ['CRIM', 'ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV','isHighValue']

data.columns = col_names
print(data.head())
print('\n')

#2. 데이터 전처리
print(data.shape) #전체 행렬확인
print(data.info()) #전체 컬럼별 데이터와 null 값이 아닌 행 확인하기
print(data.describe()) #기본 통계량 확인하기

#2-1. 결측치 처리
print(data.isnull().sum()) #컬럼별 결측치 수 확인하기
print(data.isnull().sum()/data.shape[0]) #결측치 비율 확인하기

#2-1(1) 중앙값으로 대체하기
data1 = data.copy()
med_val = data['CRIM'].median()
data1['CRIM'] = data1['CRIM'].fillna(med_val)
print(data1.info())

#2-1(2) 결측치 제거하기
data = data.loc[data['CRIM'].notnull(), ]
print(data.describe())
print(data.shape)

#2-2. 이상치 처리
import seaborn as sns 

sns.boxenplot(data['MEDV'])  #박스플롯 사분위수로 이상치 구하기

Q1, Q3 = data['MEDV'].quantile([0.25, 0.75])
IQR = Q3-Q1
upper_bound = Q3 +1.5*IQR
lower_bound = Q1 - 1.5*IQR 

print('outlier 범위 : %.2f 초과 또는 %.2f 미만' % (upper_bound, lower_bound))
print('outlier 개수 : %.0f' % len(data[(data['MEDV']>upper_bound)|(data['MEDV']<lower_bound)]))
print('outlier 비율 : %.2f' % (len(data[(data['MEDV']>upper_bound)|(data['MEDV']<lower_bound)])/len(data)))

def get_outlier_prop(x):
    Q1, Q3 = x.quantile([0.25, 0.75])
    IQR = Q3 - Q1 
    upper_bound = Q3 +1.5*IQR
    lower_bound = Q1 - 1.5*IQR 
    outliers = x[(x>upper_bound)|(x<lower_bound)]

    return '이상치 비율 ' + str(round(100*len(outliers)/len(x), 1)) + '%' #round = 반올림 함수

print(data.apply(get_outlier_prop)) #apply = 판다스 특정 함수를 원하는 컬럼에 적용

#2-2(1) 이상치 제거
#1) IQR 값을 기준으로 MEDV 변수의 이상치를 제거

Q1, Q3 = data['MEDV'].quantile([0.25, 0.75])
IQR = Q3-Q1
upper_bound = Q3 +1.5*IQR
lower_bound = Q1 - 1.5*IQR 

data1 = data[(data['MEDV']<= upper_bound)&(data['MEDV']>= lower_bound)]
print(data1.shape)

#2) MEDV 변수 값이 45이상인 경우를 이상치로 보고 제거하기
data2 = data[~(data['MEDV']>=45)]
print(data2.shape)
print('\n')

#2-3. 변수 변환
#데이터 전처리시 변수 변환을 하는 두 가지의 경우
#1) 변수의 분포가 한쪽으로 치우치는 경우 = 왜도 변환을 통해 해결 (여기서 할 것)
#2) 변수별로 데이터의 범위 및 단위가 다를 경우 = 데이터 스케일링을 통해 해결 (모형 구축에서 실행)

# 왜도 시각화 하기
import matplotlib.pyplot as plt 

cols = data.columns 

fig, axs = plt.subplots(ncols = 5, nrows = 3, figsize = (20,10))
idx = 0

#for _row in range(3):
#    for _col in range(5):
#        if idx < len(cols):
#            sns.distplot(data[cols[idx]], ax = axs[_row][_col])
#            idx += 1

#plt.tight_layout()


#왜도 값 구하기, 일반적으로 3이상이면 치우쳐 있다고 판단한다
print(data.apply(lambda x: x.skew(), axis = 0)) 


#왜도가 큰 컬럼에 대해 변수 변환 수행
import numpy as np

data['CRIM'] = np.log1p(data['CRIM']) 
print(data['CRIM'].skew())
print('\n')

#3. 회귀 모델링
#데이터 탐색
#회귀모델의 종속변수로 MEDV 변수를 사용한다. 분류 모델의 종속변수로는 isHighVale를 사용한다.

df_r = data.drop(['isHighValue'], axis = 1)

cols = ['MEDV', 'LSTAT', 'RM', 'CHAS', 'RAD', 'TAX']
print(df_r[cols].corr()) #상관관계 행렬 구하기 (다중공선성 확인 가능)

#3-1. 분석 모형 구축
#1) 데이터 분할 = 학습 데이터와 검증 데이터로 나누는 작업 필수

from sklearn.model_selection import train_test_split

X_cols = ['LSTAT', 'PTRATIO', 'TAX', 'AGE', 'INDUS', 'CRIM']

X = df_r[X_cols].values #독립변수 값
Y = df_r['MEDV'].values #종속변수 값

X_train_r, X_test_r, Y_train_r, Y_test_r = train_test_split(X,Y,test_size=0.3, random_state=123)

#2) 데이터 스케일링 = 데이터 분할 후 데이터 스케일링 진행

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_r_scaled = scaler.fit_transform(X_train_r)
X_test_r_scaled = scaler.transform(X_test_r)

#3) 모델구축
#3)-1. 선형 회귀 모델
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(X_train_r_scaled, Y_train_r)

print(model_lr.coef_) #모델의 계수 확인 (계수의 부호가 양이면 종속변수와 해당 변수가 양의 관계, 반대면 반대), 계수값이 클수록 그 영향도가 크다
print(model_lr.intercept_) #모델의 절편 확인

#3)-2. SVM
from sklearn.svm import SVR 

model_svr = SVR()
model_svr.fit(X_train_r_scaled, Y_train_r)

#3)-3. random forest
from sklearn.ensemble import RandomForestRegressor

model_rfr = RandomForestRegressor(random_state=123)
model_rfr.fit(X_train_r_scaled, Y_train_r)

for x, val in zip(X_cols, model_rfr.feature_importances_): #모델에서 사용하는 변수 중요도 확인가능
    print(f'{x} : %.3f' %val)

#4) 분석 모형 평가
#테스트 데이터로 회귀 모델을 평가해보자 (MAE, MSE, MAPE를 사용한다.)

y_pred_lr = model_lr.predict(X_test_r_scaled)
y_pred_svr = model_svr.predict(X_test_r_scaled)
y_pred_rfr = model_rfr.predict(X_test_r_scaled)

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

print('-'*30)
print('선형 회귀 결과')
print('MAE : %.3f' % mean_absolute_error(Y_test_r, y_pred_lr))
print('MSE : %.3f' % mean_squared_error(Y_test_r, y_pred_lr))
print('MAPE : %.3f' % mean_absolute_percentage_error(Y_test_r, y_pred_lr))
print('-'*30)
print('SVM 결과')
print('MAE : %.3f' % mean_absolute_error(Y_test_r, y_pred_svr))
print('MSE : %.3f' % mean_squared_error(Y_test_r, y_pred_svr))
print('MAPE : %.3f' % mean_absolute_percentage_error(Y_test_r, y_pred_svr))
print('-'*30)
print('랜덤포레스트 결과')
print('MAE : %.3f' % mean_absolute_error(Y_test_r, y_pred_rfr))
print('MSE : %.3f' % mean_squared_error(Y_test_r, y_pred_rfr))
print('MAPE : %.3f' % mean_absolute_percentage_error(Y_test_r, y_pred_rfr))
print('-'*30)

#4. 분류 모델링
#데이터 탐색
#분류 모델에서는 종속변수로 isHighValue 변수를 사용한다.
df_c = data.drop(['MEDV'], axis = 1)

import seaborn as sns #시각화 패키지

sns.boxplot(x='isHighValue', y='LSTAT', data=df_c)
sns.kdeplot(df_c.loc[df_c['isHighValue']==1, 'LSTAT'], color='orange',fill=True)
sns.kdeplot(df_c.loc[df_c['isHighValue']==0, 'LSTAT'], color='blue',fill=True)

import numpy as np

print(df_c.groupby('isHighValue').apply(np.mean).T) #각 독립변수의 평균값 구하기

#4-1 분석 모형 구축
#데이터 분할 및 스케일링 진행
#1) 데이터 분할

from sklearn.model_selection import train_test_split

X_cols2 = ['LSTAT', 'PTRATIO', 'TAX', 'AGE', 'NOX', 'INDUS', 'CRIM']

X2 = data[X_cols2].values
Y2 = data['isHighValue'].values

X_train_c, X_test_c, Y_train_c, Y_test_c = train_test_split(X2, Y2, test_size=0.3, random_state=123)

#2) 데이터 스케일링

X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.fit_transform(X_test_c)

#3) 모델구축
#3)-1. 로지스틱 회귀

from sklearn.linear_model import LogisticRegression

model_lo = LogisticRegression()
model_lo.fit(X_train_c_scaled, Y_train_c)

print(model_lo.coef_)
print(model_lo.intercept_)

#3)-2. SVM
from sklearn.svm import SVC

model_svc = SVC(probability=True)
model_svc.fit(X_train_c_scaled, Y_train_c)

#3)-3. 랜덤 포레스트

from sklearn.ensemble import RandomForestClassifier

model_rfc = RandomForestClassifier(random_state=123)
model_rfc.fit(X_train_c_scaled, Y_train_c)

for x, val in zip(X_cols2, model_rfc.feature_importances_): #모델에서 사용하는 변수 중요도 확인가능
    print(f'{x} : %.3f' %val)

#4) 분석 모형 평가
#predict 함수를 이용하여 예측값 먼저 구하기

y_pred_lo = model_lo.predict(X_test_c_scaled)
y_pred_svc = model_svc.predict(X_test_c_scaled)
y_pred_rfc = model_rfc.predict(X_test_c_scaled)

#평가지표 구하기
from sklearn.metrics import classification_report

print('-'*60)
print('로지스틱 회귀 결과')
print(classification_report(Y_test_c, y_pred_lo, labels=[0,1]))
print('-'*60)
print('SVM 결과')
print(classification_report(Y_test_c, y_pred_svc, labels=[0,1]))
print('-'*60)
print('랜덤포레스트 결과')
print(classification_report(Y_test_c, y_pred_rfc, labels=[0,1]))
print('-'*60)

#모델 성능 비교
from sklearn.metrics import roc_auc_score

y_pred_lo2 = model_lo.predict_proba(X_test_c_scaled)[:, 1]
y_pred_svc2 = model_svc.predict_proba(X_test_c_scaled)[:, 1]
y_pred_rfc2 = model_rfc.predict_proba(X_test_c_scaled)[:, 1]

print('로지스틱 회귀 결과 : %.3f' % roc_auc_score(Y_test_c, y_pred_lo2))
print('SVM 결과 : %.3f' % roc_auc_score(Y_test_c, y_pred_svc2))
print('랜덤 포레스트 결과 : %.3f' % roc_auc_score(Y_test_c, y_pred_rfc2))

