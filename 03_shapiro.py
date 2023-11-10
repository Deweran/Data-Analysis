# 12명의 수험생이 빅데이터 분석기사 시험에서 받은 점수이다. 
# Shapiro-Wilk 검정을 사용하여 데이터가 정규 분포를 따르는지 검증하시오
# H0: 데이터는 정규 분포를 따른다.
# H1: 데이터는 정규 분포를 따르지 않는다.
# Shapiro-Wilk 검정 통계량, p-value, 검증결과를 출력하시오

from scipy.stats import shapiro 
data = [75, 83, 81, 92, 68, 77, 78, 80, 85, 95, 79, 89]

# Shapiro-Wilk 검정 수행
statistic, p_value = shapiro(data)

# 결과 출력
print("Shapiro-Wilk 검정 통계량:", statistic)
print("p-value:", p_value)

# 유의 수준 0.05에서의 검정 결과 확인
alpha = 0.05
if p_value > alpha:
    print("귀무 가설을 기각할 수 없다. 데이터는 정규 분포를 따름")
else:
    print("귀무 가설을 기각한다. 데이터는 정규 분포를 따르지 않음")