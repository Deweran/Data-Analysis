# 독립표본 검정 (Independent)

# 독립 표본 t검정의 경우 집단의 정규성에 따라 접근방식이 다르다
# 정규성 검정은 shapiro , anderson(샘플 5000개 이상) 을 통해 확인



# 데이터가 정규성을 가지지 않는 경우(비모수적 검정)
# 두개 학급의 시험성적에 대한 데이터이다. 
# 두 학습의 시험 평균(비모수검정의 경우 중위값)은 동일하다 말할 수 있는지 확인 하라
import pandas as pd 
import matplotlib.pyplot as plt
df1 = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/ind1.csv')
df2 = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/ind2.csv')

plt.hist(df1,label='df1',alpha=0.4)
plt.hist(df2,label="df2",alpha=0.4)
plt.xlabel('Score bins')
plt.ylabel('Counts')
plt.legend()
plt.show()


from scipy.stats import shapiro
print(shapiro(df1)) # pvalue = 0.379
print(shapiro(df2)) # pvalue = 0.679
# 두 그룹 모두 Shapiro검정 결과 귀무가설(정규성을 가진다)을 기각 하지 못한다. 
# 두 그룹은 정규성을 가진다.

from scipy.stats import levene
print()
print(levene(df1['data'],df2['data'])) # pvalue = 0.113
# 두그룹은 levene 검정을 확인해 본결과 pvalue 는 0.11로 귀무가실을 기각히지 못한다. 
# 그러므로 등분산을 가진다

from scipy.stats import ttest_ind
print()
print(ttest_ind(df1,df2,equal_var=True)) # pvlaue = 0.006
# 등분산이기 때문에 equal_var=True 파라미터를 주고 ttest_ind 모듈을 이용하여 t test를 진행한다
# pvalue는 0.006이므로 귀무가설(각 그룹의 평균값은 동일하다)를 기각하고 대립가설을 채택한다


# 두 개 학급의 시험성적에 대한 데이터이다. 
# 두 학습의 시험 평균(비모수검정의 경우 중위값)은 동일하다 말할 수 있는지 확인 하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/scipy5.csv')

plt.hist(df['A'],alpha=0.5,label='A')
plt.hist(df['B'].dropna(),alpha=0.5,label="B")
plt.xlabel('Score bins')
plt.ylabel('Counts')
plt.legend()
plt.show()


# 데이터 분포를 확인해보니 정규성을 위해하는 것 처럼 보인다.
# 두그룹중 한 그룹만 정규성을 위배해도 독립표본 t-검정을 할 수 없다
print(shapiro(df['B'].dropna()))  # pvlaue = 0.00013
print(shapiro(df['A'])) # pvlaue = 6.17
# 두 그룹 모두 Shapiro검정 결과 귀무가설(정규성을 가진다)을 기각한다. 
# 정규성을 위배한다. 그러므로 비모수 검정을 실시해야한다.

from scipy.stats import mannwhitneyu , ranksums
print()
print(mannwhitneyu(df['A'],df['B'].dropna()))  # pvlaue = 0.98(?)
print(ranksums(df['A'],df['B'].dropna())) # pvlaue = 0.98
# Mann-Whitney U Test 검정 결과 pvalue는 0.49값으로 귀무가설(평균은같다)를 기각 할 수 없다. 
# 두그 룹의 평균은 동일하다 말할 수 있다. 
# 윌콕슨 순위합 검정(ranksums)으로 확인 해봐도 같은 결과가 나온다.


# 두 개 그룹에 대한 수치형 데이터이다. 두 그룹의 평균은 동일하다 말할 수 있는지 검정하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/ind3.csv')

plt.hist(df[df['group'] =='a'].data,label='A',alpha=0.5)
plt.hist(df[df['group'] =='b'].data,label="B",alpha=0.5)
plt.xlabel('Score bins')
plt.ylabel('Counts')
plt.legend()
plt.show()

a = df[df['group'] =='a'].data
b = df[df['group'] =='b'].data


from scipy.stats import shapiro
print(shapiro(a))
print(shapiro(b))
print("두 그룹 모두 Shapiro검정 결과 귀무가설(정규성을 가진다)을 기각 하지 못한다. \
      두 그룹은 정규성을 가진다.")

from scipy.stats import levene
print()
print(levene(a,b))
print("두그룹은 levene 검정을 확인해 본결과 pvalue 는 0.013로 귀무가실을 기각하고 대립가설을 채택한다. \
      두 그룹은 등분산이 아니다")

from scipy.stats import ttest_ind
print()
print(ttest_ind(a,b,equal_var=False))
print('''등분산이 아니기 때문에 equal_var=False 파라미터를 주고 ttest_ind 모듈을 이용하여 t test를 진행한다
pvalue는 0.02이므로 귀무가설(각 그룹의 평균값은 동일하다)를 기각하고 대립가설을 채택한다
결론적으로 두 그룹은 모두 정규성을 가지지만 등분산은 아니며 평균은 동일하다고 보기 어렵다
''')


# 두 개 그룹에 대한 수치형 데이터이다. 두 그룹의 평균은 동일하다 말할 수 있는지 검정하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/ind6.csv')

plt.hist(df['a'],alpha=0.5,label='A')
plt.hist(df['b'],alpha=0.5,label="B")
plt.xlabel('Score bins')
plt.ylabel('Counts')
plt.legend()
plt.show()

a = df['a'].dropna()
b = df['b'].dropna()


from scipy.stats import shapiro
print(shapiro(a))
print(shapiro(b))
print("두 그룹 모두 Shapiro검정 결과 귀무가설(정규성을 가진다)을 기각 하지 못한다. \
      두 그룹은 정규성을 가진다.")

from scipy.stats import levene
print()
print(levene(a,b))
print("두그룹은 levene 검정을 확인해 본결과 pvalue 는 0.047로 귀무가실을 기각하고 대립가설을 채택한다. \
      두 그룹은 등분산이 아니다")

from scipy.stats import ttest_ind
print()
print(ttest_ind(a,b,equal_var=False))
print('''등분산이 아니기 때문에 equal_var=False 파라미터를 주고 ttest_ind 모듈을 이용하여 t test를 진행한다
pvalue는 0.99이므로 귀무가설(각 그룹의 평균값은 동일하다)를 기각하기 어렵다
결론적으로 두 그룹은 모두 정규성을 가지지만 등분산은 아니며 평균은 동일하다고 볼 수 있다
''')


# 대응표본 t-검정 (paired)

# 특정 질병 집단의 투약 전후의 혈류량 변화를 나타낸 데이터이다. 투약 전후의 변화가 있는지 검정하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/rel2.csv')

fig ,ax = plt.subplots(1,2)
ax[0].boxplot(df['before'])
ax[1].boxplot(df['after'])
ax[0].set_xticklabels(['before'])
ax[1].set_xticklabels(['after'])
ax[0].set_ylim(100,350)
ax[1].set_ylim(100,350)
ax[1].get_yaxis().set_visible(False)
ax[0].set_ylabel('value')
plt.show()

from scipy.stats import shapiro
before = df['before']
after = df['after']
print(shapiro(before))
print(shapiro(after))

from scipy.stats import levene
print()
print(levene(before,after))

from scipy.stats import ttest_rel
print(ttest_rel(before,after))

# 정규성 가짐 , 등분산성 가짐 -> 대응표본의 경우 등분산성이 파라미터에 영향을 주지않음, 
# 대응표본 t 검정 결과 pvalue는 0.01로 유의수준 5%내에서 귀무가설을 기각한다 (전 후 평균은 같지 않다)


# 특정 질병 집단의 투약 전후의 혈류량 변화를 나타낸 데이터이다. 투약 전후의 변화가 있는지 검정하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/rel3.csv')

fig ,ax = plt.subplots(1,2)
ax[0].boxplot(df['before'])
ax[1].boxplot(df['after'])
ax[0].set_xticklabels(['before'])
ax[1].set_xticklabels(['after'])
ax[0].set_ylim(130,300)
ax[1].set_ylim(130,300)
ax[1].get_yaxis().set_visible(False)
ax[0].set_ylabel('value')
plt.show()

from scipy.stats import shapiro
before = df['before']
after = df['after']
print(shapiro(before))
print(shapiro(after))

from scipy.stats import levene
print()
print(levene(before,after))

from scipy.stats import ttest_rel
print(ttest_rel(before,after))
print()
# 정규성 가짐 , 등분산성 가짐 -> 대응표본의 경우 등분산성이 파라미터에 영향을 주지않음, 
# 대응표본 t 검정 결과 pvalue는 0.85로 유의수준 5%내에서 귀무가설을 기각할 수 없다 (전 후 평균은 같다)


# 특정 집단의 학습 전후 시험 성적 변화를 나타낸 데이터이다. 시험 전과 후에 차이가 있는지 검정하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/rel1.csv')

fig ,ax = plt.subplots(1,2)
ax[0].boxplot(df['before'])
ax[1].boxplot(df['after'])
ax[0].set_xticklabels(['before'])
ax[1].set_xticklabels(['after'])
ax[0].set_ylim(145,170)
ax[1].set_ylim(145,170)
ax[1].get_yaxis().set_visible(False)
ax[0].set_ylabel('value')
plt.show()

from scipy.stats import shapiro
before = df['before']
after = df['after']
print(shapiro(before))
print(shapiro(after))

from scipy.stats import levene
print()
print(levene(before,after))

from scipy.stats import ttest_rel
print(ttest_rel(before,after))
print()

from scipy.stats import wilcoxon
print(wilcoxon(before,after))
# 정규성을 가지지 않음 , 등분산성 가짐 -> 대응표본의 경우 등분산성이 파라미터에 영향을 주지않음, 
# 정규성을 가지지 않으므로 대응 표본 검정중 비모수 검정인 윌콕슨 부호순위 검정을 진행해야한다 (scipy.stats.wilcoxon)
# t-test의 경우 전후 변화에 대한 귀무가설을 기각되지만 윌콕슨 부호순위 검정을 통해서 확인해봤을때 귀무가설을 기각할 수 없다


# 한 기계 부품의 rpm 수치를 두가지 다른 상황에서 측정했다 (총 70세트) 
# b 상황이 a 상황보다 rpm값이 높다고 말할 수 있는지 검정하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/rel4.csv')

fig ,ax = plt.subplots(1,2)
ax[0].boxplot(df[df['group']=='a'].rpm)
ax[1].boxplot(df[df['group']=='b'].rpm)
ax[0].set_xticklabels(['a'])
ax[1].set_xticklabels(['b'])
ax[0].set_ylim(430,600)
ax[1].set_ylim(430,600)
ax[1].get_yaxis().set_visible(False)
ax[0].set_ylabel('rpm')
plt.show()

from scipy.stats import shapiro
a = df[df['group']=='a'].rpm
b =  df[df['group']=='b'].rpm
print(shapiro(a))
print(shapiro(b))

from scipy.stats import levene
print()
print(levene(a,b))

from scipy.stats import ttest_rel
print(ttest_rel(a,b,alternative='greater'))
print()
# 정규성을 가짐 , 등분산성 가짐 -> 대응표본의 경우 등분산성이 파라미터에 영향을 주지않음, 
# a,b,alternative='greater' 의 의미는 a >b가 대립가설이 된다는 것이다. p-value는 0.96으로
# 귀무가설인 a<=b를 기각하지 못한다. 그러므로 b상황이 a 상황보다 rpm 값이 크다고 이야기 할수 있다.


# 카이제곱 검정 (교차분석)

# 일원 카이제곱검정 (chisquare , 카이제곱 적합도 검정)
#    = 한 개의 요인에 의해 k개의 범주를 가질때 이론적 분포를 따르는지 검정

# 이원 카이제곱검정 (chi2_contingency ,fisher_exact(빈도수 5개 이하 셀이 20% 이상일때), 카이제곱독립검정)
#    = 모집단이 두개의 변수에 의해 범주화 되었을 때, 두 변수들 사이의 관계가 독립인지 아닌지 검정


# 144회 주사위를 던졌을때, 각 눈금별로 나온 횟수를 나타낸다. 
# 이 데이터는 주사위의 분포에서 나올 가능성이 있는지 검정하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/dice.csv')

plt.bar(df.dice_number,df.counts)
plt.xlabel('dice value')
plt.ylabel('counts')
plt.show()

# 주사위 눈금의 발생확률은 1/6으로 모두 동일하다. 
# 그러므로 각 눈금의 기댓값은 실제 발생한 모든값을 6으로 나눈 값이다.

from scipy.stats import chisquare
df['expected'] = (df['counts'].sum()/6).astype('int')
print(chisquare(df.counts,df.expected)) 

# p-value는 0.8로 귀무가설인 "각 주사위 눈금 발생비율은 동일함"을 기각 할 수 없다 


# 다음 데이터는 어떤 집단의 왼손잡이, 오른손 잡이의 숫자를 나타낸다. 
# 인간의 왼손잡이와 오른손잡이의 비율을 0.2:0.8로 알려져있다.
# 이 집단에서 왼손과 오른손 잡이의 비율이 적합한지 검정하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/hands2.csv')
df.head()

# 데이터에서 
target = df.hands.value_counts().to_frame()
target['expected'] = [int(target.hands.sum()*0.8),int(target.hands.sum()*0.2)]
print(target)

from scipy.stats import chisquare
print(chisquare(target.hands,target.expected))

# 알려진 비율로 계산된 기댓값을 구하여 카이제곱검정을 시행한다.
# p-value는 0.02로 유의수준 5%이내에서 귀무가설을 기각하고 대립가설을 채택한다
# 즉 주어진 집단의 왼손, 오른손 비율은 0.2, 0.8으로 볼 수 없다


# 다음 데이터는 국민 기초체력을 조사한 데이터이다. 성별과 등급이 독립적인지 검정하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/body/body.csv')
df.head()

cdf = pd.crosstab(df['측정회원성별'],df['등급'])
print(cdf)

from scipy.stats import chi2_contingency
print(chi2_contingency(cdf))
chi2 , p ,dof, expected = chi2_contingency(cdf)
print(p)
# p-value는 0에 근접하므로 측정회원성별 - 등급은 연관이 없다는 귀무가설을 기각하고, 
# 성별과 체력 등급간에는 관련이 있다고 볼 수 있다.


# 성별에 따른 동아리 활동 참석 비율을 나타낸 데이터이다. 성별과 참석간에 관련이 있는지 검정하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/scipy/fe2.csv',index_col=0)
print(df)

cdf = df.iloc[:-1,:-1]
print(cdf)

from scipy.stats import chi2_contingency,fisher_exact
print(chi2_contingency(cdf))
chi2 , p ,dof, expected = chi2_contingency(cdf)
print(p)

# 카이 제곱 검정시 p-value는 0.07로 귀무가설을 기각하지 못한다. 성별과 참석여부는 관련이 없다(독립이다).
# 하지만 5보다 작은 셀이 20%가 넘어가므로(75%) 피셔의 정확검정을 사용 해야한다.
# 피셔의 정확검정시 0.03의 값을 가지므로 귀무가설을 기각한다. 성별과 참석여부는 관련이 있다. (독립이 아니다)) 
print(fisher_exact(cdf))
