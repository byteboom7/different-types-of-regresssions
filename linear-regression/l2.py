from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import numpy as np

data = fetch_california_housing(as_frame=True)
df = data.frame

X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']].values
y = df['MedHouseVal'].values

X_mean = np.mean(X, axis=0)
X_std_dev = np.std(X, axis=0)
X_std = (X - X_mean) / X_std_dev

y_min = y.min()
y_max = y.max()
y_std = (y - y_min) / (y_max - y_min)

def calculate_parameters(epochs):
    m, n = X_std.shape
    w = np.zeros(n)
    b = 0.0
    a = 0.1
    e = 1e-7
    j = []

    for epoch in range(epochs):
        y_pred = X_std @ w + b
        error = y_pred - y_std

        dw = (1 / m) * (X_std.T @ error) + 2 * e * w
        db = (1 / m) * np.sum(error)

        w = w - a * dw
        b = b - a * db

        J = (1 / (2 * m)) * np.sum(error ** 2) + e * np.sum(w ** 2)
        j.append(J)

    y_pred_final = X_std @ w + b
    mse = np.mean((y_pred_final - y_std) ** 2)
    r2 = 1 - (np.sum((y_pred_final - y_std) ** 2) / np.sum((y_std - np.mean(y_std)) ** 2))

    print(y_pred_final[:5])
    print(y_std[:5])

    return w, b, j, J, mse, r2

w, b, j, J, mse, r2 = calculate_parameters(epochs=500)

print(f'J: {J}\nw: {w}\nb: {b}\nMSE: {mse}\nAccuracy (R^2): {r2}')

plt.plot(range(500), j)
plt.xlabel('Epochs')
plt.ylabel('Cost J')
plt.show()
print(w)