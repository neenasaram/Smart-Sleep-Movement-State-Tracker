import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import base64
import io


st.image("innomatics_logo.png", use_container_width=True)

# Function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# background image
add_bg_from_local("bg.jpg") 

# ----------- Feature Engineering Function -----------
def create_features(data):
    window_size = 60
    data['anglez_mean'] = data.groupby('series_id')['anglez'].rolling(window_size, min_periods=1).mean().reset_index(0, drop=True)
    data['anglez_std'] = data.groupby('series_id')['anglez'].rolling(window_size, min_periods=1).std().reset_index(0, drop=True)
    data['enmo_mean'] = data.groupby('series_id')['enmo'].rolling(window_size, min_periods=1).mean().reset_index(0, drop=True)
    data['enmo_std'] = data.groupby('series_id')['enmo'].rolling(window_size, min_periods=1).std().reset_index(0, drop=True)
    return data.fillna(0)

# ----------- Train Model Once -----------
selected_features = ['anglez_mean', 'anglez_std', 'enmo_mean', 'enmo_std']

@st.cache_resource
def train_model():
    train_series = pd.read_csv("train_series_processed.csv")
    train_events = pd.read_csv("train_events_processed.csv")

    train_data = train_series.merge(train_events[['series_id', 'step', 'event']], on=['series_id', 'step'], how='left')
    train_data['event'] = train_data['event'].fillna('none')
    train_data = create_features(train_data)

    X = train_data[selected_features]
    y = train_data['event'].map({'none': 0, 'onset': 1, 'wakeup': 2})

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_pca, y_train)

    return model, scaler, pca

# ----------- Streamlit App UI -----------
st.title("SleepSense: Smart Sleep Movement & State Tracker")

st.markdown("""
This app detects your **sleep state** by entered accelerometer data.  
👉 Use the ENMO calculator below to get movement intensity.
""")

# Train model only once
model, scaler, pca = train_model()

# -------- ENMO Calculator Section --------
st.subheader("📐 ENMO Calculator (Optional)")

st.markdown("""
**ENMO (Euclidean Norm Minus One)** measures movement intensity from X, Y, Z values.

| Axis    | Movement Direction           |
|---------|------------------------------|
| X-axis  | Left ↔ Right                 |
| Y-axis  | Forward ↔ Backward           |
| Z-axis  | Up ↕ Down (gravity-related)  |
""")

with st.expander("🔢 Enter X, Y, Z to calculate ENMO"):
    x = st.number_input("X-axis (Left ↔ Right)", value=0.0)
    y = st.number_input("Y-axis (Forward ↔ Backward)", value=0.0)
    z = st.number_input("Z-axis (Up ↕ Down)", value=1.0)  # Usually includes gravity
    if st.button("Calculate ENMO"):
        enmo_val = max(np.sqrt(x**2 + y**2 + z**2) - 1, 0)  # Subtract gravity
        st.success(f"✅ ENMO (Movement Intensity): {enmo_val:.3f}")

# -------- Manual Input for Prediction --------
st.subheader("📝 Enter Data for Sleep Prediction")

series_id = st.text_input("Series ID", value="001")
time_measured_sec = st.number_input("Time Measured (sec)", min_value=0)
anglez = st.number_input("Angle During Sleep (°)", value=0, format="%d")  # Integer
enmo = st.number_input("Movement Intensity During Sleep (ENMO)", value=0, format="%d")  # Integer

# Session state to collect data
if 'data_collected' not in st.session_state:
    st.session_state.data_collected = pd.DataFrame(columns=[
        'Series ID', 'Time Measured (sec)', 'Angle During Sleep (°)', 'Movement Intensity (ENMO)', 'Predicted Sleep State'
    ])

# -------- Prediction --------
if st.button("🎯 Predict Sleep State"):
    input_data = pd.DataFrame({
        'series_id': [series_id],
        'step': [time_measured_sec],
        'anglez': [anglez],
        'enmo': [enmo]
    })

    input_data = create_features(input_data)
    X_input = input_data[selected_features]
    X_input_scaled = scaler.transform(X_input)
    X_input_pca = pca.transform(X_input_scaled)

    prediction = model.predict(X_input_pca)[0]
    prediction_label = {0: 'None (No Event)', 1: 'Sleep Onset', 2: 'Wakeup'}[prediction]

    st.success(f"🧠 Predicted Sleep State: *{prediction_label.upper()}*")

    # Add prediction to collected data
    new_data = pd.DataFrame({
        'Series ID': [series_id],
        'Time Measured (sec)': [time_measured_sec],
        'Angle During Sleep (°)': [anglez],
        'Movement Intensity (ENMO)': [enmo],
        'Predicted Sleep State': [prediction_label]
    })
    st.session_state.data_collected = pd.concat([st.session_state.data_collected, new_data], ignore_index=True)
    st.success("✅ Data added to table with prediction!")

# -------- Show Collected Data --------
st.subheader("📋 Collected Data")
st.write(st.session_state.data_collected)

# -------- Save to Excel --------


if st.button("💾 Save to Excel"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        st.session_state.data_collected.to_excel(writer, index=False, sheet_name='Sleep Data')
    st.download_button(
        label="📥 Download Excel File",
        data=output.getvalue(),
        file_name="collected_sleep_data_with_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
