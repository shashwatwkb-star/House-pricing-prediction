# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ==========================================
# 2. LOAD DATASET
# ==========================================
df = pd.read_csv("train.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================================
# 3. HANDLE MISSING VALUES
# ==========================================
# Numeric → median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical → mode
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# ==========================================
# 4. OUTLIER REMOVAL (IQR METHOD)
# ==========================================
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) | 
          (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

print("After outlier removal:", df.shape)

# ==========================================
# 5. FEATURE ENGINEERING
# ==========================================
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

# ==========================================
# 6. ENCODING CATEGORICAL VARIABLES
# ==========================================
df = pd.get_dummies(df, drop_first=True)

# ==========================================
# 7. SPLIT FEATURES & TARGET
# ==========================================
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# ==========================================
# 8. TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 9. SCALING (ONLY FOR LINEAR MODELS)
# ==========================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 10. MODELS
# ==========================================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),

    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )
}

# ==========================================
# 11. TRAIN & EVALUATE MODELS
# ==========================================
results = []

for name, model in models.items():

    # Linear models use scaled data
    if name in ["Linear Regression", "Ridge", "Lasso"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append((name, mae, rmse, r2))

# ==========================================
# 12. DISPLAY RESULTS
# ==========================================
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
print(results_df.sort_values(by="RMSE"))

# ==========================================
# 13. HYPERPARAMETER TUNING (XGBOOST)
# ==========================================
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.05]
}

grid = GridSearchCV(
    XGBRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)

grid.fit(X_train, y_train)

print("\nBest XGBoost Params:", grid.best_params_)

# ==========================================
# 14. FEATURE IMPORTANCE (BEST MODEL)
# ==========================================
best_model = grid.best_estimator_

import matplotlib.pyplot as plt

importance = best_model.feature_importances_
features = X.columns

# Show top 10 features
indices = np.argsort(importance)[-10:]

plt.figure()
plt.barh(range(len(indices)), importance[indices])
plt.yticks(range(len(indices)), features[indices])
plt.title("Top 10 Important Features")
plt.show()
