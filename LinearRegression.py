import numpy as np
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx)*(y[i] - my)
    return d

x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

# 변수들의 평균
mx = np.mean(x)
my = np.mean(y)

# (x-x의 평균)제곱의 합
divisor = sum([(i - mx)**2 for i in x])


dividend = top(x, mx, y, my)

a = dividend / divisor

b = my - (a * mx)

print("x의 평균 : ", mx)
print("y의 평균", my)
print("분모 : ", divisor)
print("분자", dividend)
print("기울기 : ", a)
print("y의 절편", b)