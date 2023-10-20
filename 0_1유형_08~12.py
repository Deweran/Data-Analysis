import pandas as pd
import numpy as np
import datetime 

# 대한민국 체력장 데이터
df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/body/body.csv')
print(df.head(3))

# 1. 전체데이터의 수축기혈압(최고) - 이완기혈압(최저)의 평균을 구하여라
answer = df['수축기혈압(최고) : mmHg'].mean() - df['이완기혈압(최저) : mmHg'].mean()
print(answer)

# 2. 50~59세의 신장평균을 구하여라
def age(x):
    if 50 <= x <= 59:
        return True
    
target = df[df['측정나이'].map(age) == True]
answer = target['신장 : cm'].mean()
print(answer)

# 3. 연령대 (20~29 : 20대 …) 별 인원수를 구하여라
def stratify(x):
    if 10 <= x <= 19:
        return '10'
    elif 20 <= x <= 29:
        return '20'
    elif 30 <= x <= 39:
        return '30'
    elif 40 <= x <= 49:
        return '40'
    elif 50 <= x <= 59:
        return '50'
    else:
        return '60 ~'
    
df['class'] = df['측정나이'].map(stratify)
answer = df.groupby(['class']).size()
print(answer)

# 4. 연령대 (20~29 : 20대 …) 별 등급의 숫자를 데이터 프레임으로 표현하라
answer = df.value_counts(['class','등급']).to_frame().sort_values('class')
print(answer)

# 5. 남성 중 A등급과 D등급의 체지방률 평균의 차이(큰 값에서 작은 값의 차)를 구하여라
target = df[df.측정회원성별 == 'M']
gg = target.groupby('등급')['체지방율 : %'].mean()
answer = gg.values[3] - gg.values[0]
print(answer)

# 6. 여성 중 A등급과 D등급의 체중의 평균의 차이(큰 값에서 작은 값의 차)를 구하여라
target = df[df.측정회원성별 == 'F']
gg = target.groupby('등급')['체중 : kg'].mean()
answer = gg.values[3] - gg.values[0]
print(answer)

# 7. bmi는 자신의 몸무게(kg)를 키의 제곱(m)으로 나눈값이다. 
# 데이터의 bmi 를 구한 새로운 컬럼을 만들고 남성의 bmi 평균을 구하여라
df['bmi'] = (df['체중 : kg'])/((df['신장 : cm']/100) ** 2)
answer = df[df.측정회원성별 == 'M']['bmi'].mean()
print(answer)

# 8. bmi보다 체지방율이 높은 사람들의 체중평균을 구하여라
target = df[df['체지방율 : %'] > df.bmi]
answer = target['체중 : kg'].mean()
print(answer)

# 9. 남성과 여성의 악력 평균의 차이를 구하여라
tt = df.groupby('측정회원성별')['악력D : kg'].mean()
answer = tt.values[1] - tt.values[0]
print(answer)

# 10. 남성과 여성의 교차윗몸일으키기 횟수의 평균의 차이를 구하여라
tt = df.groupby('측정회원성별')['교차윗몸일으키기 : 회'].mean()
answer = tt.values[1] - tt.values[0]
print(answer)


# 기온 강수량 데이터
df2 = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/weather/weather2.csv")
print(df2.head(3))

# 1. 여름철(6월,7월,8월) 이화동이 수영동보다 높은 기온을 가진 시간대는 몇개인가?
df2.time = pd.to_datetime(df2.time)
target = df2[(df2.time.dt.month.isin([6,7,8])) & (df2.이화동기온 > df2.수영동기온)]
print(len(target))

# 2. 이화동과 수영동의 최대강수량의 시간대를 각각 구하여라
ihwa = df2[df2.loc[:, '이화동강수'] == df2.이화동강수.max()].iloc[0,0]
suyoung = df2[df2.loc[:, '수영동강수'] == df2.수영동강수.max()].iloc[0,0]
print(ihwa, suyoung)


# 서비스 이탈예측 데이터
df3 = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churn/train.csv")
print(df3.head(3))

# 1. 남성 이탈(Exited)이 가장 많은 국가(Geography)는 어디이고 이탈 인원은 몇명인가?
target = df3[(df3.Gender == 'Male') & (df3.Exited == 1)]
answer = target.groupby(['Geography']).size().sort_values(ascending = False).head(1)
print(answer)

# 2. 카드를 소유(HasCrCard == 1)하고 있으면서 활성멤버(IsActiveMember == 1)인 
# 고객들의 평균 나이를 소숫점 이하 4자리까지 구하여라
target = df3[(df3.HasCrCard == 1) & (df3.IsActiveMember == 1)].Age.mean()
answer = round(target, 4)
print(answer)

# 3. Balance 값이 중간값 이상을 가지는 고객들의 CreditScore의 표준편차를 소숫점이하 3자리까지 구하여라
con = df3.Balance.median()
target = df3[df3.Balance >= con].CreditScore.std()
answer = round(target,3)
print(answer)


# 성인 건강검진 데이터
df4 = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/smoke/train.csv")
print(df4.head(3))
#print(df4.columns)

# 1. 수축기혈압과 이완기혈압 수치의 차이를 새로운 컬럼(‘혈압차’) 으로 생성하고, 
# 연령대 코드별 각 그룹 중 ‘혈압차’ 의 분산이 5번째로 큰 연령대 코드를 구하여라
df4['혈압차'] = df4['수축기혈압'] - df4['이완기혈압']
answer = df4.groupby(['연령대코드(5세단위)'])['혈압차'].var().sort_values(ascending = False).index[4]
print(answer)

# 2. 비만도를 나타내는 지표인 WHtR는 허리둘레 / 키로 표현한다. 
# 일반적으로 0.58이상이면 비만으로 분류한다. 데이터중 WHtR 지표상 비만인 인원의 남/여 비율을 구하여라
df4['WHtR'] = df4['허리둘레'] / df4['신장(5Cm단위)']
target = df4[df4.WHtR >= 0.58].groupby(['성별코드']).size()
answer = target.values[1]/target.values[0]
print(answer)


# 자동차 보험가입 예측데이터
df5 = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/insurance/train.csv")
print(df5.head(3))

# 1. Vehicle_Age 값이 2년 이상인 사람들만 필터링 하고 그중에서
# Annual_Premium 값이 전체 데이터의 중간값 이상인 사람들을 찾고, 그들의 Vintage값의 평균을 구하여라
print(df5.Vehicle_Age.unique())
target = df5[df5.Vehicle_Age == '> 2 Years']
target2 = target[target.Annual_Premium >= df5.Annual_Premium.median()]
answer = target2.Vintage.mean()
print(answer)

# 2. vehicle_age에 따른 각 성별(gender)그룹의 Annual_Premium값의 평균을 구하여 아래 테이블과 동일하게 구현하라
target = df5.groupby(['Vehicle_Age','Gender'], as_index = False)['Annual_Premium'].mean()
answer = target.pivot(index='Vehicle_Age', columns='Gender', values='Annual_Premium')
print(answer)
