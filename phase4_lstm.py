import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================
# 1. LOAD & CLEAN THE V3 DATASET
# ==========================================
print("🚀 Loading V3 Master Dataset for Deep Learning...")
df = pd.read_csv('data/processed/master_hab_dataset_v3.csv')

# Drop optical proxies to prevent target leakage
cols_to_drop = ['Chl_A_Proxy', 'NDWI_Shallow_Index', 'Turbidity_NDTI']
df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Fill any remaining minor gaps with column means (safety net)
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())

# ==========================================
# 2. SEQUENCE GENERATION (THE LSTM MAGIC)
# ==========================================
print("🔄 Reshaping data into 3D time-series sequences...")

# Define how many time steps to look back (e.g., 3 steps = ~24 days of memory)
LOOKBACK = 3 

features = ['Surface_Temp', 'Air_Temp', 'Dewpoint_Temp', 'Rainfall_mm', 
            'Wind_Speed', 'Solar_Radiation', 'Evaporation', 'Runoff', 
            'Dynamic_Soil_Moisture', 'Static_Soil_Clay', 'Static_Soil_pH']
target = 'Target_NDCI'

# Ensure columns exist before extracting
features = [f for f in features if f in df_clean.columns]

# Scale the data (Neural Networks require values between 0 and 1 to converge properly)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

df_clean[features] = scaler_X.fit_transform(df_clean[features])
df_clean[[target]] = scaler_y.fit_transform(df_clean[[target]])

X_list, y_list = [], []

# Group by lake so we don't accidentally mix timelines from different dams
for lake, lake_df in df_clean.groupby('Lake_Name'):
    # Sort chronologically
    lake_df = lake_df.sort_values(by=['Year', 'Month'])
    
    # Create sequences
    X_lake = lake_df[features].values
    y_lake = lake_df[target].values
    
    for i in range(len(lake_df) - LOOKBACK):
        X_list.append(X_lake[i:(i + LOOKBACK)])
        y_list.append(y_lake[i + LOOKBACK])

X = np.array(X_list)
y = np.array(y_list)

print(f"✅ Generated {len(X)} sequences. Shape: {X.shape}")

# Train/Test Split (80% Train, 20% Test)
# We do a sequential split, not a random split, to preserve the flow of time
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==========================================
# 3. BUILD THE LSTM NEURAL NETWORK
# ==========================================
print("\n🧠 Constructing LSTM Architecture...")
model = Sequential()

# Input layer looking at our sequences
model.add(LSTM(64, return_sequences=True, input_shape=(LOOKBACK, len(features))))
model.add(Dropout(0.2)) # Prevents overfitting (memorization)

# Hidden memory layer
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))

# Output layers forecasting the exact NDCI density
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ==========================================
# 4. TRAIN THE MODEL
# ==========================================
print("\n⚡ Training Deep Learning Model (Epochs = 50)...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Predict and Inverse Scale (turn back into real-world NDCI numbers)
preds_scaled = model.predict(X_test)
preds_real = scaler_y.inverse_transform(preds_scaled)
y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Metrics
lstm_r2 = r2_score(y_test_real, preds_real)
lstm_rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))

print("\n==========================================")
print("📊 LSTM FORECASTING PERFORMANCE:")
print(f"R-squared: {lstm_r2:.4f}")
print(f"RMSE:      {lstm_rmse:.4f}")
print("==========================================")

# ==========================================
# 5. VISUALIZE LEARNING CURVE
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', color='teal', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='crimson', linewidth=2)
plt.title('LSTM Neural Network: Learning Curve', fontweight='bold', fontsize=14)
plt.xlabel('Epoch (Training Iterations)')
plt.ylabel('Mean Squared Error (Loss)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

os.makedirs('data/processed/presentation_graphs', exist_ok=True)
plt.savefig('data/processed/presentation_graphs/10_LSTM_Learning_Curve.png', dpi=300)
print("\n✅ Learning curve saved to: data/processed/presentation_graphs/10_LSTM_Learning_Curve.png")