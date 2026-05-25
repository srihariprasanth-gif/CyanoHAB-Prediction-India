import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv('data/processed/master_hab_dataset.csv')

# 2. DROP DEAD COLUMNS & MULTICOLLINEAR TARGETS
# Dropping the 100% missing columns, plus Chl_A_Proxy (since NDCI is our primary target)
cols_to_drop = ['Solar_Radiation', 'Evaporation', 'Runoff', 'Chl_A_Proxy', 'Lake_Name']
df_clean = df.drop(columns=cols_to_drop)

# 3. KNN IMPUTATION (Fill the 12-82 missing rows)
print("🩹 Healing missing data with KNN Imputation...")
imputer = KNNImputer(n_neighbors=5)
# Impute only numerical columns
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])

# 4. TRAIN/TEST SPLIT
X = df_clean.drop(columns=['Target_NDCI'])
y = df_clean['Target_NDCI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. BASELINE XGBOOST TRAINING
print("🧠 Training Baseline XGBoost Model...")
xgb_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# 6. EVALUATION
predictions = xgb_model.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"\n📊 BASELINE RESULTS:")
print(f"R-squared: {r2:.4f}")
print(f"RMSE:      {rmse:.4f}")

# 7. FEATURE IMPORTANCE CHART
importance = xgb_model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='teal')
plt.title('XGBoost Feature Importance: Drivers of Cyanobacteria (30 Dams)')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('data/processed/xgboost_feature_importance.png', dpi=300)
print("\n✅ Feature Importance chart saved to: data/processed/xgboost_feature_importance.png")