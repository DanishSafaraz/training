import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# MQTT Configuration (DIKOMMENTARI DULU UNTUK DEPLOYMENT)
MQTT_BROKER = "48be83e63863499c87afce855025c93e.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764992629489"
MQTT_PASSWORD = ".09yUhd13*nZF?A#rjKT"

# Topics
MQTT_TOPIC_SUBSCRIBE = "iot/sensor/data"
MQTT_TOPIC_PUBLISH = "iot/predict/ml"
MQTT_TOPIC_CONTROL = "iot/control/ml"

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="IoT ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .status-connected {
        color: #2ECC71;
        font-weight: bold;
    }
    .status-disconnected {
        color: #E74C3C;
        font-weight: bold;
    }
    .prediction-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==================== INIT SESSION STATE ====================
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = []
if 'mqtt_connected' not in st.session_state:
    st.session_state.mqtt_connected = False
if 'last_mqtt_msg' not in st.session_state:
    st.session_state.last_mqtt_msg = None

# ==================== SIMULATED FUNCTIONS ====================
def load_all_models():
    """Simulasi load models - untuk deployment tanpa joblib"""
    st.session_state.ml_models = {
        'Decision Tree': {'type': 'Classifier', 'accuracy': 0.85},
        'KNN': {'type': 'Classifier', 'accuracy': 0.82},
        'Logistic Regression': {'type': 'Classifier', 'accuracy': 0.80}
    }
    return True

def make_prediction_local(temperature, humidity):
    """Simulasi prediksi tanpa model asli"""
    
    if not st.session_state.ml_models:
        return {}
    
    now = datetime.now()
    hour = now.hour
    
    # Simple prediction logic
    predictions = {}
    
    for model_name, model_info in st.session_state.ml_models.items():
        # Simulate different predictions for different models
        if 'Decision' in model_name:
            if temperature < 22:
                label = 'DINGIN'
                confidence = 0.85
            elif temperature > 26:
                label = 'PANAS'
                confidence = 0.90
            else:
                label = 'NORMAL'
                confidence = 0.95
        elif 'KNN' in model_name:
            if temperature < 21:
                label = 'COLD'
                confidence = 0.80
            elif temperature > 27:
                label = 'HOT'
                confidence = 0.85
            else:
                label = 'NORMAL'
                confidence = 0.88
        else:
            if temperature < 23:
                label = 'LOW'
                confidence = 0.75
            elif temperature > 25:
                label = 'HIGH'
                confidence = 0.78
            else:
                label = 'MEDIUM'
                confidence = 0.82
        
        predictions[model_name] = {
            'label': label,
            'confidence': confidence,
            'model_type': model_info['type']
        }
    
    # Save to history
    history_entry = {
        'timestamp': datetime.now(),
        'temperature': temperature,
        'humidity': humidity,
        'predictions': predictions,
        'type': 'simulated'
    }
    st.session_state.predictions.append(history_entry)
    
    if len(st.session_state.predictions) > 50:
        st.session_state.predictions = st.session_state.predictions[-50:]
    
    return predictions

def generate_sample_sensor_data():
    """Generate sample sensor data untuk demo"""
    import random
    
    for i in range(10):
        temp = random.uniform(20, 30)
        hum = random.uniform(40, 80)
        
        st.session_state.sensor_data.append({
            'timestamp': datetime.now() - pd.Timedelta(minutes=10-i),
            'temperature': temp,
            'humidity': hum,
            'source': 'demo'
        })

# ==================== SIDEBAR ====================
def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
        
        # Connection Status
        st.markdown("---")
        st.subheader("📡 Connection Status")
        
        if st.session_state.mqtt_connected:
            st.markdown('<p class="status-connected">✅ Connected to HiveMQ</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-disconnected">❌ MQTT Disconnected</p>', unsafe_allow_html=True)
            st.info("MQTT disabled for deployment")
        
        # Model Management
        st.markdown("---")
        st.subheader("🤖 Model Management")
        
        if st.button("🔄 Load Demo Models", use_container_width=True):
            with st.spinner("Loading demo models..."):
                if load_all_models():
                    st.success(f"✅ Loaded {len(st.session_state.ml_models)} demo models")
                    st.rerun()
        
        if st.session_state.ml_models:
            st.write(f"**Models Loaded:** {len(st.session_state.ml_models)}")
            for name in st.session_state.ml_models.keys():
                st.caption(f"• {name}")
        
        # Demo Controls
        st.markdown("---")
        st.subheader("🎯 Demo Controls")
        
        temp = st.slider("Temperature (°C)", 15.0, 35.0, 25.0, 0.5)
        hum = st.slider("Humidity (%)", 30.0, 90.0, 65.0, 1.0)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 Predict", use_container_width=True):
                make_prediction_local(temp, hum)
                st.rerun()
        
        with col2:
            if st.button("📊 Generate Data", use_container_width=True):
                generate_sample_sensor_data()
                st.rerun()
        
        if st.button("📡 Simulate MQTT", use_container_width=True):
            # Simulate MQTT message
            st.session_state.last_mqtt_msg = {
                'timestamp': datetime.now(),
                'topic': MQTT_TOPIC_SUBSCRIBE,
                'data': {
                    'temperature': round(np.random.uniform(20, 30), 2),
                    'humidity': round(np.random.uniform(40, 80), 2),
                    'device_id': 'ESP32_DEMO'
                }
            }
            st.session_state.sensor_data.append({
                'timestamp': datetime.now(),
                'temperature': st.session_state.last_mqtt_msg['data']['temperature'],
                'humidity': st.session_state.last_mqtt_msg['data']['humidity'],
                'source': 'simulated_mqtt'
            })
            st.rerun()
        
        # Stats
        st.markdown("---")
        st.subheader("📊 Statistics")
        st.write(f"**Sensor Data:** {len(st.session_state.sensor_data)}")
        st.write(f"**Predictions:** {len(st.session_state.predictions)}")
        
        if st.session_state.last_mqtt_msg:
            last_time = st.session_state.last_mqtt_msg['timestamp'].strftime('%H:%M:%S')
            st.write(f"**Last Msg:** {last_time}")

# ==================== MAIN DASHBOARD ====================
def main():
    # Header
    st.markdown("<h1 class='main-title'>🤖 IoT ML Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #666;'>Streamlit Cloud Deployment - Demo Mode</h4>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Render sidebar
    render_sidebar()
    
    # Row 1: Status Cards
    st.subheader("📊 Dashboard Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.mqtt_connected:
            st.success("📡 MQTT: Connected")
        else:
            st.info("📡 MQTT: Demo Mode")
    
    with col2:
        st.metric("🤖 Models", len(st.session_state.ml_models))
    
    with col3:
        if st.session_state.sensor_data:
            latest_temp = st.session_state.sensor_data[-1]['temperature']
            st.metric("🌡️ Temp", f"{latest_temp:.1f}°C")
        else:
            st.metric("🌡️ Temp", "N/A")
    
    with col4:
        if st.session_state.sensor_data:
            latest_hum = st.session_state.sensor_data[-1]['humidity']
            st.metric("💧 Humid", f"{latest_hum:.1f}%")
        else:
            st.metric("💧 Humid", "N/A")
    
    st.markdown("---")
    
    # Row 2: Sensor Data
    st.subheader("📈 Sensor Data")
    
    if st.session_state.sensor_data:
        sensor_df = pd.DataFrame(st.session_state.sensor_data)
        
        # Tampilkan data dalam tabel
        st.dataframe(
            sensor_df.tail(10).sort_values('timestamp', ascending=False),
            use_container_width=True,
            column_config={
                'timestamp': st.column_config.DatetimeColumn(format="HH:mm:ss"),
                'temperature': st.column_config.NumberColumn(format="%.1f °C"),
                'humidity': st.column_config.NumberColumn(format="%.1f %")
            }
        )
        
        # Chart sederhana menggunakan st.line_chart
        if len(sensor_df) > 1:
            st.subheader("📊 Temperature Trend")
            chart_data = pd.DataFrame({
                'Time': sensor_df['timestamp'].dt.strftime('%H:%M'),
                'Temperature': sensor_df['temperature']
            }).set_index('Time')
            st.line_chart(chart_data)
            
            st.subheader("📊 Humidity Trend")
            chart_data = pd.DataFrame({
                'Time': sensor_df['timestamp'].dt.strftime('%H:%M'),
                'Humidity': sensor_df['humidity']
            }).set_index('Time')
            st.line_chart(chart_data)
    else:
        st.info("📭 No sensor data yet. Click 'Generate Data' in sidebar.")
    
    st.markdown("---")
    
    # Row 3: Predictions
    st.subheader("🔮 Prediction Results")
    
    if st.session_state.predictions:
        latest_pred = st.session_state.predictions[-1]
        
        if latest_pred['predictions']:
            # Display predictions in columns
            pred_items = list(latest_pred['predictions'].items())
            cols = st.columns(len(pred_items))
            
            for idx, (model_name, pred_info) in enumerate(pred_items):
                with cols[idx]:
                    # Color based on label
                    label = pred_info['label'].upper()
                    if 'DINGIN' in label or 'COLD' in label or 'LOW' in label:
                        color = '#3498DB'
                    elif 'PANAS' in label or 'HOT' in label or 'HIGH' in label:
                        color = '#E74C3C'
                    else:
                        color = '#2ECC71'
                    
                    st.markdown(f"""
                    <div class="prediction-card" style="border-left-color: {color};">
                        <h4 style="color: {color};">{model_name}</h4>
                        <h2 style="color: {color}; text-align: center;">{pred_info['label']}</h2>
                        <p style="text-align: center;">Confidence: {pred_info['confidence']:.1%}</p>
                        <p style="text-align: center; color: #666; font-size: 0.9em;">
                            Type: {pred_info['model_type']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Prediction History Table
        with st.expander("📜 View Prediction History"):
            history_data = []
            for pred in st.session_state.predictions[-10:]:
                row = {
                    'Time': pred['timestamp'].strftime('%H:%M:%S'),
                    'Temp': f"{pred['temperature']:.1f}°C",
                    'Humid': f"{pred['humidity']:.1f}%",
                    'Type': pred['type']
                }
                
                # Add predictions for each model
                for model_name in st.session_state.ml_models.keys():
                    if model_name in pred['predictions']:
                        row[model_name] = pred['predictions'][model_name]['label']
                    else:
                        row[model_name] = 'N/A'
                
                history_data.append(row)
            
            if history_data:
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions yet. Load models and make predictions.")
    
    # Row 4: Latest MQTT Message
    if st.session_state.last_mqtt_msg:
        st.markdown("---")
        st.subheader("📨 Latest Simulated MQTT Message")
        
        with st.expander("View Message Details"):
            msg = st.session_state.last_mqtt_msg
            st.write(f"**Topic:** {msg['topic']}")
            st.write(f"**Time:** {msg['timestamp'].strftime('%H:%M:%S')}")
            st.json(msg['data'])
    
    # Row 5: Quick Actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Add Random Data", use_container_width=True):
            st.session_state.sensor_data.append({
                'timestamp': datetime.now(),
                'temperature': round(np.random.uniform(20, 30), 2),
                'humidity': round(np.random.uniform(40, 80), 2),
                'source': 'random'
            })
            st.rerun()
    
    with col2:
        if st.session_state.sensor_data:
            latest = st.session_state.sensor_data[-1]
            if st.button("🧠 Predict Latest", use_container_width=True):
                make_prediction_local(latest['temperature'], latest['humidity'])
                st.rerun()
    
    with col3:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.sensor_data = []
            st.session_state.predictions = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
        <p><strong>🚀 Deployed on Streamlit Cloud - Demo Mode</strong></p>
        <p>⚠️ <strong>Note:</strong> Running in demo mode without MQTT connection</p>
        <p>For full functionality, run locally with <code>pip install joblib paho-mqtt scikit-learn</code></p>
        <p>🕐 Last update: {datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
