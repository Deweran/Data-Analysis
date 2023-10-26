import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 정규성 검정
# 다음 데이터의 정규성을 검증하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/normal1.csv')
plt.hist(df)
plt.show()

from scipy.stats import shapiro 
print(shapiro(df))
# 샤피로 검정시 p-value가 0.34이므로 유의수준 5%에서 
# 귀무가설("데이터는 정규성을 가진다")을 기각할 수 없다


# 다음 데이터의 정규성을 검증하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/normal3.csv')
plt.hist(df)
plt.show()

print(shapiro(df))
# 샤피로 검정시 p-value가 2.3e-16 이므로 유의수준 5%에서 
# 귀무가설인 "데이터는 정규성을 가진다"를 기각하고 대립가설을 채택한다
# 데이터는 정규성을 가지지 않는다


# 위의 데이터를 log변환 한 후에 정규성을 가지는지 확인하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/normal3.csv')
log_y_data = np.log1p(df)

plt.hist(log_y_data)
plt.show()

print(shapiro(log_y_data))
# 샤피로 검정시 p-value가 0.17이므로 유의수준 5%에서 
# 귀무가설("데이터는 정규성을 가진다")을 기각할 수 없다


# 다음 데이터의 정규성을 검증하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/normal6.csv')
plt.hist(df)
plt.show()




print(shapiro(df))
# 샤피로 검정시 p-value가 0.15 이므로 유의수준 5%에서 
# 귀무가설("데이터는 정규성을 가진다")을 기각할 수 없다.
# 하지만 경고 메세지에서도 보이듯이 5000개 초과의 샘플에 대해서는 
# 샤피로 검정은 정확하지 않을 수 있다.

from scipy.stats import anderson
# anderson 검정을 실시한다
print(anderson(df['data'].values))

# anderson 검정 결과의 의미는 아래 링크에서 확인 가능
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html 
# significance_level는 유의 확률값을 나타내며 critical_values는 각 유의 확률값의 기준점이 된다.
# 5%유의 수준에서 검정을 진행하려면 statistic값인 0.82이 significance_level 이 5.에 위치한 인덱스를 
# critical_values값에서 비교하면 된다. 
# 그 값은 0.786이므로 이보다 큰 0.82을 가지므로 
# 귀무가설을 기각하고 대립가설을 채택한다 -> 데이터는 정규성을 가지지 않는다고 판단한다. 
# (p-value와 기각기준 부등호 개념이 반대)


# 단일 표본 T-검정 (one-sample)
# 100명의 키 정보가 들어 있는 데이터가 있다. 데이터가 정규성을 만족하는지 확인하라.
# 그리고 평균키를 165라 판단할 수 있는지 귀무가설과 대립가설을 설정한 후 유의수준 5%로 검정하라.
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/height1.csv')

from scipy.stats import ttest_1samp
from scipy.stats import shapiro 
from scipy.stats import wilcoxon

# 정규성 검정 샤피로
static, pvalue = shapiro(df)
print('샤피로 정규성 검정 p-value : ',pvalue,'이므로')

if pvalue < 0.05:
    print('귀무가설을 기각한다. 정규성을 만족하지 않으므로 비모수 검정을 진행한다. 
    윌콕슨 순위 부호 검정을 진행한다.\n')
    print('윌콕슨 순위 부호 검정의 귀무가설은 "100명 키의 평균은 165이다." 이며 
    대립가설은 "100명 키의 평균은 165가 아니다." 이다')
    
    #윌콕슨 부호순위 검정
    static, pvalue = wilcoxon(df['height']-165) # or static, pvalue = wilcoxon(df['height'], np.ones(len(df)) *165)
    
    if pvalue < 0.05:
        print(f'검정 결과 pvalue는 {pvalue}로 결과는 귀무가설을 기각하고 대립가설을 채택한다.')
    else:
        print(f'검정 결과 pvalue는 {pvalue}로 결과는 귀무가설을 기각하지 않는다.')
    
else:
    print('귀무가설을 기각하지 않는다. 정규성을 만족하므로 단일표본 검정으로 확인한다.\n')
    print('단일표본 t-test의 귀무가설은 "100명 키의 평균은 165이다." 이며 
    대립가설은 "100명 키의 평균은 165가 아니다." 이다')
    
    #단일 표본 t 검정
    static, pvalue = ttest_1samp(df['height'],165) 
    if pvalue < 0.05:
        print(f'검정 결과 pvalue는 {pvalue}로 결과는 귀무가설을 기각하고 대립가설을 채택한다.')
    else:
        print(f'검정 결과 pvalue는 {pvalue}로 결과는 귀무가설을 기각하지 않는다.')  


# 100명의 키 정보가 들어 있는 데이터가 있다. 데이터가 정규성을 만족하는지 확인하라.
# 그리고 평균키를 165라 판단할 수 있는지 귀무가설과 대립가설을 설정한 후 유의수준 5%로 검정하라.
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/height2.csv')

# 정규성 검정 샤피로
static, pvalue = shapiro(df)
print('샤피로 정규성 검정 p-value : ',pvalue,'이므로')

if pvalue < 0.05:
    print('귀무가설을 기각한다. 정규성을 만족하지 않으므로 비모수 검정을 진행한다. 
    윌콕슨 순위 부호 검정을 진행한다.\n')
    print('윌콕슨 순위 부호 검정의 귀무가설은 "100명 키의 평균은 165이다." 이며 
    대립가설은 "100명 키의 평균은 165가 아니다." 이다')
    
    #윌콕슨 부호순위 검정
    static, pvalue = wilcoxon(df['height']-165) # or static, pvalue = wilcoxon(df['height'], np.ones(len(df)) *165)
    
    if pvalue < 0.05:
        print(f'검정 결과 pvalue는 {pvalue}로 결과는 귀무가설을 기각하고 대립가설을 채택한다.')
    else:
        print(f'검정 결과 pvalue는 {pvalue}로 결과는 귀무가설을 기각하지 않는다.')
    
else:
    print('귀무가설을 기각하지 않는다. 정규성을 만족하므로 단일표본 검정으로 확인한다.\n')
    print('단일표본 t-test의 귀무가설은 "100명 키의 평균은 165이다." 이며 
    대립가설은 "100명 키의 평균은 165가 아니다." 이다')
    
    #단일 표본 t 검정
    static, pvalue = ttest_1samp(df['height'],165) 
    if pvalue < 0.05:
        print(f'검정 결과 pvalue는 {pvalue}로 결과는 귀무가설을 기각하고 대립가설을 채택한다.')
    else:
        print(f'검정 결과 pvalue는 {pvalue}로 결과는 귀무가설을 기각하지 않는다.')  



# 등분산 검정
# 두 개 학급의 시험성적에 대한 데이터이다 그룹간 등분산 검정을 시행하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/scipy2.csv')
print(df.head(3))

from scipy.stats import bartlett
from scipy.stats import fligner
from scipy.stats import levene

a = df[df['class'] =='A'].score
b = df[df['class'] =='B'].score

print(bartlett(a,b))

print(fligner(a,b,center='median')) #default
print(fligner(a,b,center='mean')) 

print(levene(a,b, center='median')) #default
print(levene(a,b,center='mean'))


# 등분산검정의 방법은 3가지가 있다. 
# pvalue값은 5% 유의수준이라면 0.05보다 작은 경우 "각 그룹은 등분산이다"라는 귀무가설을 기각한다
# 아래의 결과를 보면 모두 0.05보다 크므로 귀무가설을 기각할수 없음을 알 수 있다.


# 두 개 학급의 시험성적에 대한 데이터이다 그룹간 등분산 검정을 시행하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/scipy3.csv')
print(df.head(3))

a = df[df['class'] =='A'].score
b = df[df['class'] =='B'].score

print(bartlett(a,b))
print()
print(fligner(a,b,center='median')) #default
print(fligner(a,b,center='mean')) 

print(levene(a,b, center='median')) #default
print(levene(a,b,center='mean'))

# bartlett 검정 결과 pvalue는 0.05보다 크고
# fligner, levene 검정 결과 pvalue는 0.05보다 작다. 
# fligner, levene는 bartlett보다 좀 더 robust하다는 특징이 있다.
# 어떤 검정의 결과를 사용해야하는지는 정해지지 않았지만 상황에 따라 특징들을 서술할 수 있다면 문제 없지 않을까...


# 두 개 학급의 시험성적에 대한 데이터이다 그룹간 등분산 검정을 시행하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/scipy6.csv')
print(df.head(3))

print(bartlett(df.A, df.B))
print(fligner(df.A, df.B))
print(levene(df.A, df.B))

# BartlettResult -> 등분산이다  // FlignerResult , LeveneResult -> 등분산이 아니다


# 두 개 학급의 시험성적에 대한 데이터이다 그룹간 등분산 검정을 시행하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/scipy5.csv')
print(df.head(3))

print(bartlett(df.A, df.B))
print(bartlett(df.A, df.B.dropna()))
print()

print(fligner(df.A, df.B))
print(fligner(df.A, df.B.dropna()))
print()

print(levene(df.A, df.B))
print(levene(df.A, df.B.dropna()))

# bartlett ,fligner 두 검정은 nan값을 지우고 사용해야한다. 
# LeveneResult의 경우 nan값이 포함된다면 연산이 제대로 안된다
