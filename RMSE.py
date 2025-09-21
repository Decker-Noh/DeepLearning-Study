import numpy as np
# 임의값
ab = [3, 76]

data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x= [i[0] for i in data]
y= [i[1] for i in data]

def predict(x):
    return ab[0]*x + ab[1]

def rmse(p, a):
    return np.sqrt(((p-a)**2).mean())

def rmse_val(predict_result, y):
    return rmse(np.array(predict_result), np.array(y))

predict_result = []

for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("실제x : %.f, 실제y : %.f, 예측y : %.f" % (x[i], y[i], predict(x[i])))

print("최종 : ", str(rmse_val(predict_result, y)))