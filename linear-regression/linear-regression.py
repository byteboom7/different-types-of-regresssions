from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import numpy as np

data = fetch_california_housing(as_frame=True)
df = data.frame

X = df[['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']].values
y = df['MedHouseVal'].values
X_mean = np.mean(X, axis=0)
X_std_dev = np.std(X, axis=0)
X_std = (X - X_mean) / X_std_dev
J = 0
w = np.zeros(8)
b = 0
m = 0
j = []
def calculate_parameters(epochs):
    global w,b,J,m,j
    y_min = y.min()
    y_max = y.max()
    y_std = (y - y_min) / (y_max - y_min)
    dw = np.zeros(8)
    db = 0
    a = 0.000001
    c = 0
    for epoch in range(epochs):
        J = 0
        for c in range(len(X_std)):
            dw = (((np.dot(w,X_std[c]) + b - y_std[c])*X_std[c]))
            db = w@X_std[c] + b - y_std[c]
            w = w - a*dw
            b = b - a*db
            J += (1/2*((w@X_std[c]+b - y_std[c])**2))
        j.append(J)
    print((X_std @ w + b)[:5])
    print(y_std[:5])
    return f'J: {J}, w: {w}, b: {b} \n j: {j}, MSE: {np.mean((X_std @ w + b- y)**2)}, Accuracy (Using R^2): {1 - (np.sum((X_std @ w + b- y_std)**2))/np.sum((y_std - np.mean(y_std))**2)}'
   


print(calculate_parameters(epochs=500))

"""plt.scatter(X_std, y, alpha=0.5)
plt.plot(X_std, w@X + b, color='red')
plt.xlabel('bmi')
plt.ylabel('target')
plt.show()"""
plt.plot(range(500),j)
plt.show()
print(w)