import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Real_Combine.csv')
print("Dataset loaded successfully")
print(df.head())

# Check null values heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Drop null values
df = df.dropna()

# Independent and dependent features
X = df.iloc[:, :-1]  # independent features
y = df.iloc[:, -1]   # dependent feature

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Pairplot
sns.pairplot(df)
plt.show()

# Correlation Heatmap
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.title("Correlation Heatmap")
plt.show()

# Feature importance using ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()
model.fit(X, y)

print("Feature Importances:", model.feature_importances_)

# Plot top 5 features
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.title("Top 5 Feature Importances")
plt.show()

# Distribution plot of target
sns.displot(y, kde=True)
plt.title("Target Distribution")
plt.show()

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Linear Regression
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)

print("R^2 on train set:", regressor.score(X_train, y_train))
print("R^2 on test set:", regressor.score(X_test, y_test))

# Cross Validation
from sklearn.model_selection import cross_val_score
score = cross_val_score(regressor, X, y, cv=5)
print("Cross Validation Scores:", score)
print("Mean CV Score:", score.mean())

# Coefficient DataFrame
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# Predictions
prediction = regressor.predict(X_test)
print("Prediction Shape:", prediction.shape)

# Plot residuals
sns.histplot(y_test - prediction, kde=True)
plt.title("Residual Distribution")
plt.show()

# Scatter plot
plt.scatter(y_test, prediction)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

# Error Metrics
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

# Save Model with Pickle
import pickle

with open('regression_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)

print("Model saved as regression_model.pkl")
