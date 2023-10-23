import pandas as pd
import numpy as np

x_test = pd.read_csv('C:\\Users\\김이슬아\\Desktop\\python\\빅분기 실기 자료\\404_x_test.csv')
x_train = pd.read_csv('C:\\Users\\김이슬아\\Desktop\python\\빅분기 실기 자료\\404_x_train.csv')
y_train = pd.read_csv('C:\\Users\\김이슬아\\Desktop\python\\빅분기 실기 자료\\404_y_train.csv')

print(x_train.info())
print(x_train.head(3))

col_del = ['ID']
obj = ['Gender','Ever_Married','Graduated','Profession','Spending_Score']
num = ['Age','Work_Experience','Family_Size']
y = ['Segmentation']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train[num])

x_train[num] = scaler.transform(x_train[num])
x_test[num] = scaler.transform(x_test[num])

from sklearn.preprocessing import LabelEncoder
x = pd.concat([x_train[obj], x_test[obj]])
le = LabelEncoder()

for col in obj:
    le = LabelEncoder()
    le.fit(x[col])
    x_train[col] = le.transform(x_train[col])
    x_test[col] = le.transform(x_test[col])

    print(col)
    print(le.classes_)
    print('\n')

from sklearn.model_selection import train_test_split 
x_tr, x_val, y_tr, y_val = train_test_split(x_train[num + obj], y_train[y].values.ravel(),
                                            test_size = 0.33, stratify = y_train[y].values.ravel())

# model
from sklearn.ensemble import RandomForestClassifier 
model_r = RandomForestClassifier(n_estimators=100, max_depth=6, random_state = 120)
model_r.fit(x_tr, y_tr)


# assessment 
from sklearn.metrics import f1_score 
model_r_a = model_r.predict(x_val)

score = f1_score(y_val, model_r_a, average = 'macro')
print('randomforest = ', score) # 0.511

# hyper parameter tuning 
# from sklearn.model_selection import GridSearchCV
# parameters={'n_estimators':[50,100],'max_depth':[4,6]}
# model3=RandomForestClassifier()
# clf=GridSearchCV(estimator=model3, param_grid=parameters, cv=3)
# clf.fit(X_train,Y_train)
# print('최적의 파라미터: ',clf.best_params_) # {'max_depth': 6,'n_estimators': 100}

# submit 
pred = model_r.predict(x_test[num + obj])
result = pd.DataFrame({'ID':x_test.ID, 'Segmentation': pred})
print(result)

# result.to_csv('000003000.csv', index = False)


