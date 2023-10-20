import pandas as pd
import numpy as np
import datetime

# 핸드폰 가격 예측데이터
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/mobile/train.csv")
print(df.head(3))
print(df.columns)

# 1. price_range 의 각 value를 그룹핑하여 각 그룹의 n_cores 의 빈도가 가장높은 value와 그 빈도수를 구하여라
answer = df[['price_range','n_cores']].groupby(['price_range','n_cores']).size().sort_values()\
    .groupby(level = 0).tail(1)
# groupby(level = ) => The level in groupby() is used when you have multiple indices and 
# you want to use only one index of the DataFrame. 즉, 여러개의 인덱스로 그룹핑했을때 특정한 하나의 인덱스를
# 지정하여 처리하고 싶을 때, 사용한다. 
print(answer)

# 2. price_range 값이 3인 그룹에서 상관관계가 2번째로 높은 두 컬럼과 그 상관계수를 구하여라
target = df[df.price_range == 3].corr().unstack().sort_values(ascending=False)
answer  = target.loc[target != 1].reset_index().iloc[1]
print(answer)


# 비행탑승 경험 만족도 데이터
df2 = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/airline/train.csv")
print(df2.head(3))

# 1. Arrival Delay in Minutes 컬럼이 결측치인 데이터들 중 
# ‘neutral or dissatisfied’ 보다 ‘satisfied’의 수가 더 높은 Class는 어디인가?
target = df2[df2['Arrival Delay in Minutes'].isnull()].groupby(['Class','satisfaction'], as_index = False).size()\
    .pivot(index = 'Class',columns = 'satisfaction')
answer = target[target['size']['neutral or dissatisfied'] < target['size']['satisfied']]
print(answer)


# 수질 음용성 여부 데이터
df3 = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/waters/train.csv")
print(df3.head(3))

# 1. ph값은 상당히 많은 결측치를 포함한다.
# 결측치를 제외한 나머지 데이터들 중 사분위값 기준 하위 25%의 값들의 평균값은?
target = df3['ph'].dropna()
answer = target.loc[target <= target.quantile(0.25)].mean()
print(answer)


# 의료 비용 예측 데이터
train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/MedicalCost/train.csv")
print(train.head(3))

# 1. 흡연자와 비흡연자 각각 charges의 상위 10% 그룹의 평균의 차이는?
high = train.loc[train.smoker =='yes'].charges.quantile(0.9)
high2 = train.loc[train.smoker == 'no'].charges.quantile(0.9)
mean_yes = train.loc[(train.smoker == 'yes') & (train.charges >= high)].charges.mean()
mean_no = train.loc[(train.smoker == 'no') & (train.charges >= high2)].charges.mean()
answer = mean_yes - mean_no
print(answer)


# 킹카운티 주거지 가격예측문제 데이터
df4 = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/kingcountyprice//train.csv")
print(df4.head(3))

# 1. bedrooms의 빈도가 가장 높은 값을 가지는 데이터들의 price의 상위 10%와 하위 10%값의 차이를 구하여라
target = df4[df4.bedrooms == df4.bedrooms.value_counts().index[0]]
answer = (target.price.quantile(0.9)) - (target.price.quantile(0.1))
print(answer)


# 대학원 입학가능성 데이터
df5 = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/admission/train.csv")
print(df5.head(3))

# 1. Serial No. 컬럼을 제외하고 ‘Chance of Admit’을 종속변수, 나머지 변수를 독립변수라 할때, 
# 랜덤포레스트를 통해 회귀 예측을 할 떄 변수중요도 값을 출력하라 (시드값에 따라 순서는 달라질수 있음)
from sklearn.ensemble import RandomForestRegressor

df_t = df5.drop([df5.columns[0]],axis=1)
x = df_t.drop([df5.columns[-1]],axis=1)
y = df_t[df5.columns[-1]]

ml = RandomForestRegressor()

ml.fit(x,y)

result = pd.DataFrame({'importance':ml.feature_importances_}, x.columns).sort_values('importance',ascending=False)
print(result)


# 레드 와인 퀄리티 예측 데이터
wine = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/redwine/train.csv")
print(wine.head(3))

# 1. quality 값이 3인 그룹과 8인 데이터그룹의 
# 각 컬럼별 독립변수의 표준편차 값의 차이를 구할때 그값이 가장 큰 컬럼명을 구하여라
three = wine[wine.quality == 3].std()
eight = wine[wine.quality == 8].std()
target = eight - three
answer = target.sort_values(ascending = False).index[0]
print(answer)

# 약물 분류 데이터
drug = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/drug/train.csv")
print(drug.head(3))

# 1. 남성들의 연령대별 (10살씩 구분 0~9세 10~19세 …) Na_to_K값의 평균값을 구해서 데이터 프레임으로 표현하여라
pre = drug.loc[drug.Sex == 'M']
pre2= pre.copy()
pre2['Age2'] = (pre.Age // 10) *10

answer = pre2.groupby('Age2')['Na_to_K'].mean().to_frame()
print(answer)
