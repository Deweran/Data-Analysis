import pandas as pd
import numpy as np

x_test = pd.read_csv('C:/Users/USER/Desktop/Python work space/Data Analysis/census_X_test.csv')
x_train = pd.read_csv('C:/Users/USER/Desktop/Python work space/Data Analysis/census_X_train.csv')
y_train = pd.read_csv('C:/Users/USER/Desktop/Python work space/Data Analysis/census_y_train.csv')

#데이터 전처리
print(x_train.info())
print(x_train.describe())

print(x_train['capital_gain'].quantile([q/20 for q in range(15, 21)]))
print(x_train['capital_loss'].quantile([q/20 for q in range(15, 21)]))

x_train['capital_gain_yn'] = np.where(x_train['capital_gain']>0, 1, 0)
x_train['capital_loss_yn'] = np.where(x_train['capital_loss']>0, 1, 0)

x_test['capital_gain_yn'] = np.where(x_test['capital_gain']>0, 1, 0)
x_test['capital_loss_yn'] = np.where(x_test['capital_loss']>0, 1, 0)

col_del =[]
col_num = ['age', 'education_num','hours_per_week','capital_gain','capital_loss']
col_cat = ['workclass','marital_status','occupation','relationship','race','sex',\
           'native_country','capital_gain_yn','capital_loss_yn']
col_y = ['target']

x_train = x_train.drop(col_del, axis= 1)
x_test = x_test.drop(col_del, axis=1)

train_df = pd.concat([x_train, y_train], axis=1)

for _col in col_num:
    print('-'*80)
    print(_col)
    print(train_df.groupby(col_y)[_col].describe(), end='\n\n')

for _col in col_cat:
    print(train_df.groupby(_col, as_index=False)[col_y].mean().sort_values(by=col_y, ascending=False)\
          , end='\n\n')

from sklearn.preprocessing import LabelEncoder

x = pd.concat([x_train, x_test])

for _col in col_cat:
    le = LabelEncoder()
    le.fit(x_train[_col])
    x_train[_col] = le.transform(x_train[_col])
    x_test[_col] = le.transform(x_test[_col])

#데이터 모형 구축
from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.3, stratify=y_train)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_tr[col_num] = scaler.fit_transform(x_tr[col_num])
x_val[col_num] = scaler.transform(x_val[col_num])
x_test[col_num] = scaler.transform(x_test[col_num])

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(x_tr, y_tr.values.ravel())

from xgboost import XGBClassifier

model_xgb1 = XGBClassifier()
model_xgb1.fit(x_tr, y_tr.values.ravel())

model_xgb2 = XGBClassifier(n_estimators=1000, learning_rate = 0.1, max_depth = 10)
model_xgb2.fit(x_tr, y_tr.values.ravel(), early_stopping_rounds=50, eval_metric='auc',\
                eval_set=[(x_val, y_val)], verbose=10)

from sklearn.metrics import roc_auc_score

y_pred_rf = model_rf.predict_proba(x_val)
y_pred_xgb1 = model_xgb1.predict_proba(x_val)

score_rf = roc_auc_score(y_val, y_pred_rf[:,1])
score_xgb1 = roc_auc_score(y_val, y_pred_xgb1[:,1])

print(score_rf)
print(score_xgb1)

pd.DataFrame({'feature' : x_tr.columns, 'fi_rf' : model_rf.feature_importances_, 'fi_xgb' :\
              model_xgb1.feature_importances_})

col_del = ['capital_gain_yn', 'capital_loss_yn']

x_tr = x_tr.drop(col_del, axis = 1)
x_val = x_val.drop(col_del, axis = 1)
x_test = x_test.drop(col_del, axis = 1)

from sklearn.model_selection import GridSearchCV

grid_params = {
    'n_estimators' : [50, 100, 200],
    'max_depth' : [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1,2,4]
}

rf_cv = GridSearchCV(estimator=model_rf, param_grid=grid_params, cv=5)
rf_cv.fit(x_train, y_train.values.ravel())

print(pd.DataFrame(rf_cv.cv_results_).head())
print(rf_cv.best_params_)

model_rf2 = RandomForestClassifier(n_estimators=50,
                                   max_depth=15,
                                   min_samples_leaf=1,
                                   min_samples_split=5)

model_rf2.fit(x_tr, y_tr.values.ravel())

y_pred_rf2 = model_rf2.predict_proba(x_val)
score_rf2 = roc_auc_score(y_val, y_pred_rf2[:,1])
print(score_rf2)

grid_params = {'max_depth':[3,5,7,10],
               'min_child_weight': [1,2],
               'consample_bytree':[0.6, 0.8],
               'subsample': [0.6,0.8]}

xgb_cv = GridSearchCV(estimator=model_xgb1, param_grid=grid_params, cv=5)
xgb_cv.fit(x_tr, y_tr.values.ravel())

print(xgb_cv.best_params_)

params = {'colsample_bytree':0.6,
          'max_depth':7,
          'min_child_weight':1,
          'subsample':0.8}

model_xgb3 = XGBClassifier(n_estimators = 1000, learning_rate=0.05)
model_xgb3.set_params(**params)

model_xgb3.fit(x_tr, y_tr, early_stopping_rounds=50, eval_metric='auc',
               eval_set=[(x_val,y_val)], verbose=10)

print(model_xgb3.best_score)

pred = model_xgb3.predict_proba(x_test)[:,1]
#pd.DataFrame({'index':x_test.index, 'target':pred}).to_csv(index=False)



