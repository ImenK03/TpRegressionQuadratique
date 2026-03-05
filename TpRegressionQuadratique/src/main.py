import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("prix_maisons.csv")

print(df.head())
print(df.dtypes)
print(df.shape)

x = df["surface"]
y = df["prix"]

mu_x = x.mean()
sigma_x = x.std()
mu_y = y.mean()
sigma_y = y.std()
print("μx =", mu_x)
print("σx =", sigma_x)
print("μy =", mu_y)
print("σy =", sigma_y)

x_std = (x - mu_x) / sigma_x
y_std = (y - mu_y) / sigma_y

print(x_std.mean(), x_std.std())
print(y_std.mean(), y_std.std())

plt.scatter(x_std, y_std)
plt.xlabel("Surface standardisée")
plt.ylabel("Prix standardisé")
plt.title("Relation surface / prix")

plt.show()

def quadratic_regression(a, b, c, x):
    return a * x**2 + b * x + c

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def rmse(y_pred, y_true):
    return np.sqrt(mse(y_pred, y_true))

def backpropagation_quadratic(a, b, c, x, y, lr):

    n = len(x)

    y_pred = a * x**2 + b * x + c

    e = y_pred - y

    dL_da = (2/n) * np.sum(e * x**2)
    dL_db = (2/n) * np.sum(e * x)
    dL_dc = (2/n) * np.sum(e)

    a = a - lr * dL_da
    b = b - lr * dL_db
    c = c - lr * dL_dc
    
    rmse = np.sqrt(np.mean((y_pred - y)**2))

    return a, b, c, rmse

def gradient_descent_quadratic(x, y, lr=0.01, epochs=500):

    a = np.random.randn() * 0.1
    b = np.random.randn() * 0.1
    c = np.random.randn() * 0.1

    rmse_history = []

    for epoch in range(epochs):

        y_pred = a * x**2 + b * x + c

        e = y_pred - y
        n = len(x)

        dL_da = (2/n) * np.sum(e * x**2)
        dL_db = (2/n) * np.sum(e * x)
        dL_dc = (2/n) * np.sum(e)

        a = a - lr * dL_da
        b = b - lr * dL_db
        c = c - lr * dL_dc

        rmse = np.sqrt(np.mean((y_pred - y)**2))
        rmse_history.append(rmse)

        if epoch % 50 == 0:
            print("epoch:", epoch, "rmse:", rmse)

    return a, b, c, rmse_history

a, b, c, rmse_history = gradient_descent_quadratic(x, y)

x_sorted = np.sort(x)
y_pred_curve = a * x_sorted**2 + b * x_sorted + c

plt.scatter(x, y)
plt.plot(x_sorted, y_pred_curve)

plt.xlabel("surface standardisée")
plt.ylabel("prix standardisé")
plt.title("Modèle quadratique")

plt.show()

plt.plot(rmse_history)

plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.title("Évolution de la RMSE")

plt.show()

def gradient_descent_linear(x, y, lr=0.01, epochs=500):

    a = np.random.randn() * 0.1
    b = np.random.randn() * 0.1

    rmse_history = []
    n = len(x)

    for epoch in range(epochs):

        y_pred = a * x + b
        e = y_pred - y

        dL_da = (2/n) * np.sum(e * x)
        dL_db = (2/n) * np.sum(e)

        a = a - lr * dL_da
        b = b - lr * dL_db

        rmse = np.sqrt(np.mean((y_pred - y)**2))
        rmse_history.append(rmse)

    return a, b, rmse_history

rmse_final_linear = rmse_history[-1]

rmse_final_quad = rmse_history_quad[-1]