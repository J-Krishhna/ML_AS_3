import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("StudentsPerformance.csv")

np.random.seed(42)
df["study_hours"] = np.random.randint(1, 8, size=len(df))
df["attendance"] = np.random.randint(60, 100, size=len(df))
df["sleep_hours"] = np.random.randint(4, 9, size=len(df))

df["parental level of education"] = df["parental level of education"].astype("category").cat.codes
df["test preparation course"] = df["test preparation course"].astype("category").cat.codes

df["final_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

features = [
    "study_hours",
    "attendance",
    "parental level of education",
    "test preparation course",
    "sleep_hours"
]

X = df[features]
y = df["final_score"]

X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

coefficients = pd.Series(model.coef_, index=features)
print(coefficients)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.05)
lasso.fit(X_train, y_train)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Predicted vs Actual Scores")
plt.show()

coefficients.sort_values().plot(kind="barh")
plt.title("Linear Regression Coefficients")
plt.show()

residuals = y_test - y_pred
plt.hist(residuals, bins=20)
plt.title("Residual Distribution")
plt.show()
