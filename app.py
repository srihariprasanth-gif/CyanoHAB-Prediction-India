import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="CyanoHAB Predictor", page_icon="🦠", layout="wide")

st.title("🌊 Cyanobacterial HAB Prediction Dashboard")
st.markdown("""
**Pan-Indian Ecological Framework** This application utilizes a biologically-constrained XGBoost architecture to forecast cyanobacterial bloom intensity (NDCI) based on catchment hydrology, non-point source nutrient loads, and regional climate biomes.
""")

# --- 2. CACHED MODEL TRAINING ---
@st.cache_resource
def load_and_train_model():
    # Load Datasets
    df_south = pd.read_csv('data/processed/master_hab_dataset_chemical.csv')
    df_pan = pd.read_csv('data/processed/pan_india_testing_dataset.csv')
    
    # Apply Biome Labels
    df_south['Climate_Zone'] = 'Tropical_Deccan'
    biome_mapping = {
        "Bhakra_Dam": "Himalayan_Alpine", "Tehri_Dam": "Himalayan_Alpine", "Pong_Dam": "Himalayan_Alpine",
        "Sardar_Sarovar": "Western_Arid", "Ukai_Dam": "Western_Arid", "Jayakwadi_Dam": "Western_Arid",
        "Indira_Sagar": "Central_Gangetic", "Gandhi_Sagar": "Central_Gangetic", "Rihand_Dam": "Central_Gangetic",
        "Hirakud_Dam": "Eastern_Plateau", "Maithon_Dam": "Eastern_Plateau", "Panchet_Dam": "Eastern_Plateau",
        "Koyna_Dam": "Tropical_Deccan", "Ujjani_Dam": "Tropical_Deccan", "Bisalpur_Dam": "Western_Arid"
    }
    df_pan['Climate_Zone'] = df_pan['Lake_Name'].map(biome_mapping)
    
    # Merge and format
    df_master = pd.concat([df_south, df_pan], ignore_index=True)
    cols_to_drop = ['Lake_Name', 'Year', 'Month', 'Chl_A_Proxy', 'NDWI_Shallow_Index', 'Turbidity_NDTI']
    X = df_master.drop(columns=[c for c in cols_to_drop if c in df_master.columns or 'Target' in c])
    y = df_master['Target_NDCI']
    
    X = X.fillna(X.mean(numeric_only=True))
    
    # One-Hot Encode and save the exact column structure
    X_encoded = pd.get_dummies(X, columns=['Climate_Zone'])
    model_columns = X_encoded.columns
    
    # Train the Model
    model = XGBRegressor(n_estimators=150, max_depth=7, learning_rate=0.08, random_state=42)
    model.fit(X_encoded, y)
    
    return model, model_columns

# Load model quietly in the background
with st.spinner("Initializing XGBoost Architecture..."):
    xgb_model, train_cols = load_and_train_model()

# --- 3. SIDEBAR: USER INPUTS ---
st.sidebar.header("🔬 Input Parameters")
st.sidebar.markdown("Adjust the environmental conditions to simulate bloom risk.")

biome_input = st.sidebar.selectbox("Geographical Biome", 
    ["Tropical_Deccan", "Himalayan_Alpine", "Western_Arid", "Central_Gangetic", "Eastern_Plateau"]
)

temp_input = st.sidebar.slider("Surface Temperature (°C)", min_value=10.0, max_value=40.0, value=28.5, step=0.5)
runoff_input = st.sidebar.slider("Catchment Runoff (mm)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
soil_input = st.sidebar.slider("Dynamic Soil Moisture (m³/m³)", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Chemical Load (ECM)")
n_load_input = st.sidebar.slider("Nitrogen Load (kg/month)", min_value=0, max_value=50000, value=2500, step=100)
p_load_input = st.sidebar.slider("Phosphorus Load (kg/month)", min_value=0, max_value=10000, value=400, step=50)

# --- 4. PREDICTION LOGIC ---
# Create a dataframe from the user inputs
input_dict = {
    'Surface_Temp': [temp_input],
    'Runoff': [runoff_input],
    'Dynamic_Soil_Moisture': [soil_input],
    'Calculated_Nitrogen_Load_kg': [n_load_input],
    'Calculated_Phosphorus_Load_kg': [p_load_input],
    'Climate_Zone': [biome_input]
}
input_df = pd.DataFrame(input_dict)

# Ensure all background features are present (filled with median/0) so the model doesn't break
for col in train_cols:
    if col not in input_df.columns and not col.startswith('Climate_Zone'):
        input_df[col] = 0 # Default fallback for unlisted physical features

# Apply One-Hot Encoding to the user input
input_encoded = pd.get_dummies(input_df, columns=['Climate_Zone'])
# Reindex to match the exact training columns, filling missing biomes with 0
input_encoded = input_encoded.reindex(columns=train_cols, fill_value=0)

# Generate Prediction
prediction = xgb_model.predict(input_encoded)[0]

# Normalize prediction to a 0-100% Risk Score (Assuming typical NDCI ranges from -0.1 to 0.4)
risk_score = min(max(((prediction + 0.1) / 0.5) * 100, 0), 100)

# --- 5. MAIN DASHBOARD UI ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Model Output: Bloom Intensity")
    st.metric(label="Predicted NDCI (Normalized Difference Chlorophyll Index)", value=f"{prediction:.4f}")
    
    st.subheader("Eutrophication Risk Level")
    st.progress(int(risk_score))
    
    if risk_score < 30:
        st.success("🟢 Low Risk: Water column is stable. Baseline algal presence.")
    elif risk_score < 70:
        st.warning("🟡 Moderate Risk: Conditions favor cyanobacterial proliferation. Monitor closely.")
    else:
        st.error("🔴 Severe Warning: Perfect storm conditions detected. High probability of toxic bloom event.")

with col2:
    st.subheader("Biological Interpretation")
    st.info(f"""
    **Current Scenario Analysis:**
    * **Biome Engine:** The model is applying baseline threshold rules for the **{biome_input.replace('_', ' ')}** region.
    * **Nutrient Driver:** An influx of **{n_load_input:,} kg** of Nitrogen acts as the primary biomass catalyst.
    * **Hydrological Transport:** Soil saturation at **{soil_input} m³/m³** combined with **{runoff_input} mm** of runoff is flushing these non-point source nutrients directly into the reservoir.
    """)