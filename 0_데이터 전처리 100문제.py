import pandas as pd
import numpy as np

# ----- Getting & Knowing Data -----
# 1. 데이터를 로드하라
data = pd.read_csv('C:\\Users\\USER\\Desktop\\Python work space\\Data Analysis\\201_boston.csv')

# 2. 데이터의 상위 2개 행을 출력하라
print(data.head(2))

# 3. 데이터의 행과 열 갯수를 파악하라
print(data.shape)
print('Row : ', data.shape[0])
print('Column : ', data.shape[1])

# 4. 전체 컬럼을 출력하라
w_column = data.columns
print(w_column)

# 5. 6번째 컬럼명을 출력하라
print(w_column[5])

# 6. 6번째 컬럼의 데이터 타입을 확인하라
print(data.iloc[:, 5].dtype)

# 7. 데이터셋의 인덱스 구성을 출력하라
print(data.index)

# 8. 6번째 컬럼의 3번째 값은 무엇인가?
answer = data.iloc[2,5]
print(answer)

# 9. 데이터를 로드하라 (원본이 한글이여서 적절한 처리가 선행되어야 한다.)
dataurl = 'https://raw.githubusercontent.com/Datamanim/pandas/main/Jeju.csv'
data_2 = pd.read_csv(dataurl, encoding='euc-kr')

print(data_2.head(2))

# 10. 데이터의 마지막 3개 행을 출력하라
print(data.tail(3))

# 11. 수치형 변수를 가진 컬럼을 출력하라
ans_f = data_2.select_dtypes(exclude=(object, int)).columns
print(ans_f)

# 12. 범주형 변수를 가진 컬럼을 출력하라
ans_o = data_2.select_dtypes(include=object).columns 
print(ans_o)

# 13. 각 컬럼의 결측치 숫자를 파악하라
ans = data_2.isnull().sum()
print(ans)

# 14. 각 컬럼의 데이터 수, 데이터 타입을 한번에 확인하라
print(data_2.info())

# 15. 각 수치형 변수의 분포 (사분위, 평균, 표준편차, 최대 , 최소)를 확인하라
print(data_2.describe())

# 15+ 수치형, 변수형 각각의 데이터 프레임에 describe를 해보면?
data_f = data_2[ans_f]
data_o = data_2[ans_o]

print(data_f.describe())
print(data_o.describe())

# 16. 거주인구 컬럼의 값들을 출력하라
residence = data_2['거주인구']
print(residence)

# 17. 평균 속도 컬럼의 4분위 범위값(IQR)을 구하여라
_4square = data_2['평균 속도'].quantile(0.75) - data_2['평균 속도'].quantile(0.25)
print(_4square)

# 18. 읍면동명 컬럼의 유일값 갯수를 출력하라
n_unique = data_2['읍면동명'].nunique()
print(n_unique)

# 19. 읍면동명 컬럼의 유일값을 모두 출력하라
unique = data_2['읍면동명'].unique()
print(unique)

# ----- Filtering & Sorting -----
# 20. 새로운 데이터를 로드하라
DataUrl = 'https://raw.githubusercontent.com/Datamanim/pandas/main/chipo.csv'
data_3 = pd.read_csv(DataUrl)

print(data_3.head(2))

# 21. quantity 값이 3인 데이터를 추출하여 첫 5행을 출력하라
ans_q = data_3['quantity'] == 3
print(data_3[ans_q].head(5))

# 22. quantity컬럼 값이 3인 데이터를 추출하여 index를 0부터 정렬하고 첫 5행을 출력하라
ans_o_q = data_3[ans_q].reset_index(drop=True)
print(ans_o_q.head(5))

# 23. quantity , item_price 두개의 컬럼으로 구성된 새로운 데이터 프레임을 정의하라
n_data = data_3[['quantity', 'item_price']]
print(n_data)

# 24. item_price 컬럼의 달러표시 문자를 제거하고 float 타입으로 저장하여 new_price 컬럼에 저장하라
n_data['new_price'] = n_data['item_price'].str[1:].astype('float')
print(n_data['new_price'].head(2))

# 24+ 데이터 프레임에 정규표현식 사용하기 
# Series(여기서는 데이터프레임 컬럼).str => 컬럼 내용을 시리즈로 변환시켜줌 (문자열로 변환시켜줌)
#                                        문자열처럼 인덱스, 슬라이싱을 사용할 수 있게 만들어준다. 
# 그러므로 replace도 사용가능해진다. 
# Series.str.replace(pat= , repl= , n= -1, case= None, flags = 0, regex = None)
# pat = 찾고자 하는 문자열이나 정규표현식 입력
# repl = 대체할 문자 입력
# n = 바꿀 갯수 설정, 디폴트시 모든 경우 바꿈
# case = 대소문자 구분할지 여부
# flags = int, 정규표현식 플래그 사용
# regex = 정규표현식인지 표현

# 25. new_price 컬럼이 5이하의 값을 가지는 데이터프레임을 추출하고, 전체 갯수를 구하여라
con = n_data['new_price'] <= 5
ans_wn = len(n_data[con])
print(ans_wn)

# 26. item_name명이 Chicken Salad Bowl 인 데이터 프레임을 추출하고 index 값을 초기화 하여라
con = data_3['item_name'] == 'Chicken Salad Bowl'
ans_c = data_3[con].reset_index(drop=True)
print(ans_c.head(5))

# 27. new_price값이 9 이하이고 item_name 값이 Chicken Salad Bowl 인 데이터 프레임을 추출하라
data_3['new_price'] = n_data['new_price']
con_1 = data_3['new_price'] <= 9 
con_2 = data_3['item_name'] == 'Chicken Salad Bowl'
ans = data_3[con_1 & con_2]
print(ans.head(5))

# 28. df의 new_price 컬럼 값에 따라 오름차순으로 정리하고 index를 초기화 하여라
ans = data_3.sort_values('new_price').reset_index(drop=True)
print(ans.head(5))

# 28+ 내림차순으로는?
ans_2 = data_3.sort_values(by = 'new_price', ascending = False).reset_index(drop=True)
print(ans_2.head(3))

# 29. df의 item_name 컬럼 값중 Chips 포함하는 경우의 데이터를 출력하라
con = data_3['item_name'].str.contains('Chips')
ans = data_3[con]
print(ans.head(3)) 

# 30. df의 짝수번째 컬럼만을 포함하는 데이터프레임을 출력하라
ans = data_3.iloc[:,::2]
print(ans.head(2))

# 31. df의 new_price 컬럼 값에 따라 내림차순으로 정리하고 index를 초기화 하여라
# #28+ 확인하기

# 32. df의 item_name 컬럼 값이 Steak Salad 또는 Bowl 인 데이터를 인덱싱하라
con_1 = data_3['item_name'] == 'Steak Salad'
con_2 = data_3['item_name'] == 'Bowl'
ans = data_3[con_1 | con_2]
print(ans.head(3))

# 33. df의 item_name 컬럼 값이 Steak Salad 또는 Bowl 인 데이터를 데이터 프레임화 한 후, 
# item_name를 기준으로 중복행이 있으면 제거하되 첫번째 케이스만 남겨라
s_data = data_3[con_1 | con_2]
s_data = s_data.drop_duplicates('item_name')
print(s_data)

# 34. df의 item_name 컬럼 값이 Steak Salad 또는 Bowl 인 데이터를 데이터 프레임화 한 후, 
# item_name를 기준으로 중복행이 있으면 제거하되 마지막 케이스만 남겨라
s_data = data_3[con_1 | con_2]
s_data = s_data.drop_duplicates('item_name', keep = 'last')
print(s_data)

# 35. df의 데이터 중 new_price값이 new_price값의 평균값 이상을 가지는 데이터들을 인덱싱하라
mean = data_3['new_price'].mean()
con = data_3['new_price'] >= mean
print(data_3[con].head(5))

# 36. df의 데이터 중 item_name의 값이 Izze 데이터를 Fizzy Lizzy로 수정하라
con = data_3['item_name'] == 'Izze'
data_3.loc[con, 'item_name'] = 'Fizzy Lizzy'
ans = data_3
print(ans.head(3))

# 37. df의 데이터 중 choice_description 값이 NaN 인 데이터의 갯수를 구하여라
ans = data_3['choice_description'].isnull().sum()
print(ans)

# 38. df의 데이터 중 choice_description 값이 NaN 인 데이터를 NoData 값으로 대체하라(loc 이용)
con = data_3['choice_description'].isnull()
data_3.loc[con, 'choice_description'] = 'NoData'
print(data_3.head(5))

# 39. df의 데이터 중 choice_description 값에 Black이 들어가는 경우를 인덱싱하라
con = data_3['choice_description'].str.contains('Black')
ans = data_3[con]
print(ans.head(3))

# 40. df의 데이터 중 choice_description 값에 Vegetables 들어가지 않는 경우의 갯수를 출력하라
con = ~(data_3['choice_description'].str.contains('Vegetables'))
ans = len(data_3[con])
print(ans)

# 41. df의 데이터 중 item_name 값이 N으로 시작하는 데이터를 모두 추출하라
con = data_3['item_name'].str.startswith('N')
ans = data_3[con]
print(ans.head(3))

# 42. df의 데이터 중 item_name 값의 단어갯수가 15개 이상인 데이터를 인덱싱하라
con = data_3['item_name'].str.len() >= 15
ans = data_3[con]
print(ans.head())

# 43. df의 데이터 중 new_price값이 lst에 해당하는 경우의 데이터 프레임을 구하고 그 갯수를 출력하라 
lst =[1.69, 2.39, 3.39, 4.45, 9.25, 10.98, 11.75, 16.98]
con = data_3['new_price'].isin(lst)
ans = len(data_3[con])
print(ans)

# ---- Grouping -----
# 44. 데이터를 로드하고 상위 5개 컬럼을 출력하라
DataUrl_2 = 'https://raw.githubusercontent.com/Datamanim/pandas/main/AB_NYC_2019.csv'
data_4 = pd.read_csv(DataUrl_2)
print(data_4.head())

# 45. 데이터의 각 host_name의 빈도수를 구하고 host_name으로 정렬하여 상위 5개를 출력하라
ans = data_4.groupby('host_name').size()
print(ans.head(5))

# 46. 데이터의 각 host_name의 빈도수를 구하고 빈도수 기준 내림차순 정렬한 데이터 프레임을 만들어라. 
# 빈도수 컬럼은 counts로 명명하라
new_df = data_4.groupby('host_name').size().to_frame().\
    rename(columns={0:'counts'}).sort_values('counts', ascending=False)
print(new_df.head(5))

# 47. neighbourhood_group의 값에 따른 neighbourhood컬럼 값의 갯수를 구하여라 
# 즉, 정렬할 컬럼이 두개인 문제
ans = data_4.groupby(['neighbourhood_group', 'neighbourhood'], as_index = False).size()
print(ans.head(4))

# 48. neighbourhood_group의 값에 따른 neighbourhood컬럼 값 중 
# neighbourhood_group그룹의 최댓값들을 출력하라
g_data = data_4.groupby(['neighbourhood_group', 'neighbourhood'], as_index = False).size()
ans = g_data.groupby('neighbourhood_group', as_index = False).max()
print(ans)

# 49. neighbourhood_group 값에 따른 price값의 평균, 분산, 최대, 최소 값을 구하여라
n_data = data_4[['neighbourhood_group', 'price']]
ans = n_data.groupby('neighbourhood_group').agg(['mean','var','max','min'])
print(ans)

# 50. neighbourhood_group 값에 따른 reviews_per_month 평균, 분산, 최대, 최소 값을 구하여라
n_data = data_4[['neighbourhood_group','reviews_per_month']]
ans = n_data.groupby('neighbourhood_group').agg(['mean','var','max','min'])
print(ans)

# 51. neighbourhood 값과 neighbourhood_group 값에 따른 price 의 평균을 구하라
n_data = data_4[['neighbourhood','neighbourhood_group','price']]
g_data = n_data.groupby(['neighbourhood','neighbourhood_group']).mean()
# or 
# g_data = data_4.groupby(['neighbourhood','neighbourhood_group'])['price].mean()
print(g_data.head())

# 52. neighbourhood 값과 neighbourhood_group 값에 따른 price 의 평균을 
# 계층적 indexing 없이 구하라
g_data = data_4.groupby(['neighbourhood','neighbourhood_group'])['price'].mean().unstack()
print(g_data.head())

# 53. neighbourhood 값과 neighbourhood_group 값에 따른 price 의 평균을 계층적 
# indexing 없이 구하고 nan 값은 -999값으로 채워라
ans = g_data.fillna(-999)
print(ans.head())

# 54. 데이터중 neighbourhood_group 값이 Queens값을 가지는 데이터들 중 neighbourhood 그룹별로 
# price값의 평균, 분산, 최대, 최소값을 구하라
con = data_4['neighbourhood_group'] == 'Queens'
ans = data_4[con].groupby('neighbourhood')['price'].agg(['mean','var','max','min'])
print(ans.head())

# 55. 데이터중 neighbourhood_group 값에 따른 room_type 컬럼의 숫자를 구하고 
# neighbourhood_group 값을 기준으로 각 값의 비율을 구하여라
ans = data_4[['neighbourhood_group','room_type']].\
    groupby(['neighbourhood_group','room_type']).size().unstack()
ans.loc[:,:] = (ans.values /ans.sum(axis=1).values.reshape(-1,1))
print(ans.head(3))

# ----- Apply & Map -----
# 56. 데이터를 로드하고 데이터 행과 열의 갯수를 출력하라
DataUrl_3 = 'https://raw.githubusercontent.com/Datamanim/pandas/main/BankChurnersUp.csv'
data_5 = pd.read_csv(DataUrl_3)
print(data_5.shape)

# 57. Income_Category의 카테고리를 map 함수를 이용하여 다음과 같이 변경하여 
# newIncome 컬럼에 매핑하라 
dic = {'Unknown' : 'N',
'Less than $40K' : 'a',
'$40K - $60K': 'b',
'$60K - $80K' : 'c',
'$80K - $120K' : 'd',
'$120K +' : 'e'}

data_5['newIncome'] = data_5['Income_Category'].map(lambda x: dic[x])
ans = data_5['newIncome']
print(ans.head())

# 58. Income_Category의 카테고리를 apply 함수를 이용하여 다음과 같이 변경하여 
# newIncome 컬럼에 매핑하라
def changeCategory(x):
    if x =='Unknown':
        return 'N'
    elif x =='Less than $40K':
        return 'a'
    elif x =='$40K - $60K':   
        return 'b'
    elif x =='$60K - $80K':    
        return 'c'
    elif x =='$80K - $120K':   
        return 'd'
    elif x =='$120K +' :     
        return 'e'

data_5['newIncome'] = data_5['Income_Category'].apply(changeCategory)

Ans = data_5['newIncome']

# 59. Customer_Age의 값을 이용하여 나이 구간을 AgeState 컬럼으로 정의하라. 
# (0~9 : 0 , 10~19 :10 , 20~29 :20 … 각 구간의 빈도수를 출력하라

data_5['AgeState']  = data_5['Customer_Age'].map(lambda x: x//10 *10)
Ans = data_5['AgeState'].value_counts().sort_index()
print(Ans)

# 60. Education_Level의 값중 Graduate단어가 포함되는 값은 1 그렇지 않은 경우에는 
# 0으로 변경하여 newEduLevel 컬럼을 정의하고 빈도수를 출력하라

def level(x):
    if 'Graduate' in x:
        return 1
    else:
        return 0

data_5['newEduLevel'] = data_5['Education_Level'].apply(level)
ans = data_5['newEduLevel'].value_counts()
print(ans)

# 61. Credit_Limit 컬럼값이 4500 이상인 경우 1 그외의 경우에는 모두 0으로 하는
# newLimit 정의하라. newLimit 각 값들의 빈도수를 출력하라

data_5['newLimit'] = data_5['Credit_Limit'].map(lambda x: 1 if x >= 4500 else 0)
ans = data_5['newLimit'].value_counts()
print(ans)

# 62. Marital_Status 컬럼값이 Married 이고 Card_Category 컬럼의 값이 Platinum인 경우 1 그외의 
# 경우에는 모두 0으로 하는 newState컬럼을 정의하라. newState의 각 값들의 빈도수를 출력하라

def check(x):
    if x.Marital_Status =='Married' and x.Card_Category =='Platinum':
        return 1
    else:
        return 0

data_5['newState'] = data_5.apply(check, axis=1)
ans = data_5['newState'].value_counts()
print(ans)

# 63. Gender 컬럼값 M인 경우 male F인 경우 female로 값을 변경하여 Gender 컬럼에 새롭게 정의하라. 
# 각 value의 빈도를 출력하라

data_5['Gender'] = data_5['Gender'].map(lambda x: 'Male' if x == 'M' else 'Female')
ans = data_5['Gender'].value_counts()
print(ans)

# ----- Time Series -----
# 64. 데이터를 로드하고 각 열의 데이터 타입을 파악하라
DataUrl_4 = 'https://raw.githubusercontent.com/Datamanim/pandas/main/timeTest.csv'
data_6 = pd.read_csv(DataUrl_4)
print(data_6.info())

# 65. Yr_Mo_Dy을 판다스에서 인식할 수 있는 datetime64타입으로 변경하라
data_6['Yr_Mo_Dy'] = pd.to_datetime(data_6['Yr_Mo_Dy'])
ans = data_6['Yr_Mo_Dy']
print(ans.head(4))

# 66. Yr_Mo_Dy에 존재하는 년도의 유일값을 모두 출력하라
ans = data_6['Yr_Mo_Dy'].dt.year.unique()
print(ans)

# 67. Yr_Mo_Dy에 년도가 2061년 이상의 경우에는 모두 잘못된 데이터이다.
# 해당경우의 값은 100을 빼서 새롭게 날짜를 Yr_Mo_Dy 컬럼에 정의하라
def fix_century(x):
    import datetime
    
    year = x.year - 100 if x.year >= 2061 else x.year
    return pd.to_datetime(datetime.date(year, x.month, x.day))

data_6['Yr_Mo_Dy'] = data_6['Yr_Mo_Dy'].apply(fix_century)
ans = data_6['Yr_Mo_Dy']
print(ans.head(5))

# 68. 년도별 각컬럼의 평균값을 구하여라
data_6['year'] = data_6['Yr_Mo_Dy'].dt.year
ans = data_6.groupby('year').mean()
#ans = data_6.groupby(data_6.Yr_Mo_Dy.dt.year).mean()
print(ans.head(4))

# 69. weekday컬럼을 만들고 요일별로 매핑하라 ( 월요일: 0 ~ 일요일 :6)
data_6['weekday'] = data_6.Yr_Mo_Dy.dt.weekday
Ans = data_6['weekday'].head(3).to_frame()
print(Ans)

# 70. weekday컬럼을 기준으로 주말이면 1 평일이면 0의 값을 가지는 WeekCheck 컬럼을 만들어라
data_6['WeekCheck'] = data_6['weekday'].map(lambda x: 1 if x in [5,6] else 0)
ans = data_6['WeekCheck']
print(ans.head(4))

# 71. 년도, 일자 상관없이 모든 컬럼의 각 달의 평균을 구하여라
ans = data_6.groupby(data_6['Yr_Mo_Dy'].dt.month).mean()
print(ans.head(3))

# 72. 모든 결측치는 컬럼기준 직전의 값으로 대체하고 첫번째 행에 결측치가 있을경우 
# 뒤에있는 값으로 대채하라
data_6 = data_6.fillna(method='ffill').fillna(method='bfill')
print(data_6.isnull().sum())

# 73. 년도 - 월을 기준으로 모든 컬럼의 평균값을 구하여라
ans = data_6.groupby(data_6['Yr_Mo_Dy'].dt.to_period('M')).mean()
print(ans.head(4))

# 74. RPT 컬럼의 값을 일자별 기준으로 1차차분하라
data_6['diff'] = data_6['RPT'].diff()
ans = data_6['diff']
print(ans.head(4))

# 75. RPT와 VAL의 컬럼을 일주일 간격으로 각각 이동평균한값을 구하여라
ans = data_6[['RPT','VAL']].rolling(7).mean()
print(ans.head(9))

# 76. 년-월-일:시 컬럼을 pandas에서 인식할 수 있는 datetime 형태로 변경하라. 
# 서울시의 제공데이터의 경우 0시가 24시로 표현된다
DataUrl_5 = 'https://raw.githubusercontent.com/Datamanim/pandas/main/seoul_pm.csv'
data_7 = pd.read_csv(DataUrl_5)

def change_date(x):
    import datetime
    hour = x.split(':')[1]
    date = x.split(":")[0]
    
    if hour =='24':
        hour ='00:00:00'
        
        FinalDate = pd.to_datetime(date +" "+ hour) + datetime.timedelta(days=1)
        
    else:
        hour = hour +':00:00'
        FinalDate = pd.to_datetime(date +" "+ hour)
    
    return FinalDate

data_7['(년-월-일:시)'] = data_7['(년-월-일:시)'].apply(change_date)
print(data_7.head(3))

# 77. 일자별 영어요일 이름을 dayName 컬럼에 저장하라
data_7['dayname'] = data_7['(년-월-일:시)'].dt.day_name()
ans = data_7['dayname']
print(ans.head(3))

# 78. 일자별 각 PM10등급의 빈도수를 파악하라
Ans1 = data_7.groupby(['dayname','PM10등급'],as_index=False).size()
Ans2 = Ans1.pivot(index='dayname',columns='PM10등급',values='size').fillna(0)

print(Ans1.head())
print(Ans2.head())

# 79. 시간이 연속적으로 존재하며 결측치가 없는지 확인하라
# 시간을 차분했을 경우 첫 값은 nan, 이후 모든 차분값이 동일하면 연속이라 판단한다.
check = len(data_7['(년-월-일:시)'].diff().unique())
if check == 2:
    Ans = True
else:
    Ans = False

print(Ans)

# 80. 오전 10시와 오후 10시(22시)의 PM10의 평균값을 각각 구하여라
con_1 = data_7['(년-월-일:시)'].dt.hour == 10
con_2 = data_7['(년-월-일:시)'].dt.hour == 22

ans_1 = data_7[con_1]['PM10'].mean()
ans_2 = data_7[con_2]['PM10'].mean()

print('오전 10시 PM10 : ' + str(ans_1), '\n' + '오후 10시 PM10 : ' + str(ans_2))

# 80+데이터 프레임으로 뽑으려면?
date = data_7['(년-월-일:시)'].dt.hour
#print(date) 시간을 int 형식으로 쭉 뽑아준다. 
ans = data_7.groupby(data_7['(년-월-일:시)'].dt.hour)['PM10'].mean().iloc[[10,22]]
print(ans.head())

# or 
ans = data_7.groupby(data_7['(년-월-일:시)'].dt.hour)['PM10'].mean().\
    iloc[[10,22]].to_frame()

# 81. 날짜 컬럼을 index로 만들어라
data_7.set_index('(년-월-일:시)', inplace = True ,drop = True)
ans = data_7
print(ans.head(3))

# 82. 데이터를 주단위로 뽑아서 최소,최대 평균, 표준표차를 구하여라
# 이미 날짜 컬럼이 인덱스화 되었을때에 사용가능한 resample 메서드
#Ans = data_7.resample('W').agg(['min','max','mean','std'])
#print(Ans.head())

# ----- Pivot -----
# 83. Indicator을 삭제하고 First Tooltip 컬럼에서 신뢰구간에 해당하는 표현을 지워라
Dataurl_6 = 'https://raw.githubusercontent.com/Datamanim/pandas/main/under5MortalityRate.csv'
data_8 = pd.read_csv(Dataurl_6)

data_8.drop(['Indicator'], axis=1, inplace=True)
data_8['First Tooltip'] = data_8['First Tooltip'].map(lambda x: float(x.split("[")[0]))
ans = data_8
print(ans.head())

# 84. 년도가 2015년 이상, Dim1이 Both sexes인 케이스만 추출하라
con_1 = data_8['Period'] >= 2015
con_2 = data_8['Dim1'] == 'Both sexes'
n_data = data_8[con_1 & con_2]
print(n_data.head(3))

# 85. 84번 문제에서 추출한 데이터로 아래와 같이 나라에 따른 년도별 사망률을 데이터 프레임화 하라
ans = n_data.pivot(index='Location', columns='Period', values='First Tooltip')
print(ans)

# 86. Dim1에 따른 년도별 사망비율의 평균을 구하라
Ans = data_8.pivot_table(index='Dim1',columns='Period',values='First Tooltip',aggfunc='mean')
print(Ans.iloc[:, :4])

# 87. 데이터에서 한국 KOR 데이터만 추출하라
dataUrl_7 ='https://raw.githubusercontent.com/Datamanim/pandas/main/winter.csv'
data_9 = pd.read_csv(dataUrl_7)

con = data_9['Country'] == 'KOR'
n_data = data_9[con]
print(n_data.head())

# 88. 한국 올림픽 메달리스트 데이터에서 년도에 따른 medal 갯수를 데이터프레임화 하라
ans = n_data.pivot_table(index='Year', columns='Medal', aggfunc='size').fillna(0)
print(ans)

# 89. 전체 데이터에서 sport종류에 따른 성별수를 구하여라
g_data = data_9.pivot_table(index='Sport', columns='Gender', aggfunc='size').fillna(0)
print(g_data)

# 90. 전체 데이터에서 Discipline종류에 따른 따른 Medal수를 구하여라
d_data = data_9.pivot_table(index='Discipline', columns='Medal', aggfunc='size').fillna(0)
print(d_data)

# ----- Merge & Concat -----
# 91. df1과 df2 데이터를 하나의 데이터 프레임으로 합쳐라
Dataurl_8 = 'https://raw.githubusercontent.com/Datamanim/pandas/main/mergeTEst.csv'
data_10 = pd.read_csv(Dataurl_8,index_col= 0)

df1 = data_10.iloc[:4,:]
df2 = data_10.iloc[4:,:]

total = pd.concat([df1,df2])
print(total)

# 92. df3과 df4 데이터를 하나의 데이터 프레임으로 합쳐라. 
# 둘다 포함하고 있는 년도에 대해서만 고려한다
df3 = data_10.iloc[:2,:4]
df4 = data_10.iloc[5:,3:]

total = pd.concat([df3,df4], join='inner')
print(total)

# 93. df3과 df4 데이터를 하나의 데이터 프레임으로 합쳐라. 
# 모든 컬럼을 포함하고, 결측치는 0으로 대체한다
Ans = pd.concat([df3,df4],join='outer').fillna(0)
print(Ans.head())

# 94. df5과 df6 데이터를 하나의 데이터 프레임으로 merge함수를 이용하여 합쳐라. 
# Algeria컬럼을 key로 하고 두 데이터 모두 포함하는 데이터만 출력하라
df5 = data_10.T.iloc[:7,:3]
df6 = data_10.T.iloc[6:,2:5]

Ans = pd.merge(df5,df6,on='Algeria',how='inner')
print(Ans)

# 95. df5과 df6 데이터를 하나의 데이터 프레임으로 merge함수를 이용하여 합쳐라. 
# Algeria컬럼을 key로 하고 합집합으로 합쳐라
ans = pd.merge(df5,df6,on='Algeria',how='outer')
print(ans)

