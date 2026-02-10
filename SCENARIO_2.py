print("Jaya Krishna G - 24BAD042")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("auto-mpg.csv")

df.replace("?", np.nan, inplace=True)
df["horsepower"] = df["horsepower"].astype(float)
df["horsepower"] = df["horsepower"].fillna(df["horsepower"].mean())


X = df[["horsepower"]].values
y = df["mpg"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

degrees = [2, 3, 4]
results = {}

plt.scatter(X, y)

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    y_train_pred = model.predict(X_poly_train)
    y_test_pred = model.predict(X_poly_test)

    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)

    results[d] = (mse, rmse, r2)

    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_plot_scaled = scaler.transform(X_plot)
    X_plot_poly = poly.transform(X_plot_scaled)
    y_plot = model.predict(X_plot_poly)

    plt.plot(X_plot, y_plot, label=f"Degree {d}")

plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.title("Polynomial Regression Curve Fitting")
plt.show()

for d, (mse, rmse, r2) in results.items():
    print(f"Degree {d} -> MSE: {mse}, RMSE: {rmse}, R2: {r2}")

ridge = Ridge(alpha=1.0)
poly = PolynomialFeatures(degree=4)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

ridge.fit(X_poly_train, y_train)
y_ridge_pred = ridge.predict(X_poly_test)

print("Ridge Regression")
print("MSE:", mean_squared_error(y_test, y_ridge_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_ridge_pred)))
print("R2:", r2_score(y_test, y_ridge_pred))

