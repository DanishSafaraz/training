import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import joblib
import os
import json
import time
import paho.mqtt.client as mqtt
import ssl
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# MQTT Configuration
MQTT_BROKER = "48be83e63863499c87afce855025c93e.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764992629489"
MQTT_PASSWORD = ".09yUhd13*nZF?A#rjKT"

# Topics
MQTT_TOPIC_SUBSCRIBE = "iot/sensor/data"      # Untuk terima data sensor
MQTT_TOPIC_PUBLISH = "iot/predict/ml"         # Untuk trigger predict
MQTT_TOPIC_CONTROL = "iot/control/ml"         # Untuk kontrol Arduino

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
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = []
if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None
if 'mqtt_connected' not in st.session_state:
    st.session_state.mqtt_connected = False
if 'last_mqtt_msg' not in st.session_state:
    st.session_state.last_mqtt_msg = None

# ==================== MQTT FUNCTIONS ====================
def setup_mqtt():
    """Setup MQTT client dengan callback"""
    
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            st.session_state.mqtt_connected = True
            client.subscribe(MQTT_TOPIC_SUBSCRIBE)
            st.success(f"✅ Subscribed to {MQTT_TOPIC_SUBSCRIBE}")
        else:
            st.error(f"❌ MQTT Connection failed: {rc}")
    
    def on_message(client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            data = json.loads(payload)
            
            # Simpan message terakhir
            st.session_state.last_mqtt_msg = {
                'timestamp': datetime.now(),
                'topic': msg.topic,
                'data': data
            }
            
            # Jika ada data sensor, simpan
            if 'temperature' in data and 'humidity' in data:
                sensor_entry = {
                    'timestamp': datetime.now(),
                    'temperature': float(data['temperature']),
                    'humidity': float(data['humidity']),
                    'source': 'MQTT'
                }
                st.session_state.sensor_data.append(sensor_entry)
                
                # Keep only last 100 entries
                if len(st.session_state.sensor_data) > 100:
                    st.session_state.sensor_data = st.session_state.sensor_data[-100:]
            
            # Trigger rerun untuk update UI
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing MQTT message: {e}")
    
    try:
        client = mqtt.Client(client_id=f"dashboard_{int(time.time())}")
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        
        # SSL setup
        client.tls_set(cert_reqs=ssl.CERT_NONE)
        client.tls_insecure_set(True)
        
        # Set callbacks
        client.on_connect = on_connect
        client.on_message = on_message
        
        # Connect
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        
        st.session_state.mqtt_client = client
        
        # Tunggu koneksi
        time.sleep(2)
        
        return True
        
    except Exception as e:
        st.error(f"MQTT Setup error: {str(e)}")
        return False

def publish_mqtt(topic, message):
    """Publish message ke MQTT topic"""
    if st.session_state.mqtt_client and st.session_state.mqtt_connected:
        try:
            result = st.session_state.mqtt_client.publish(topic, json.dumps(message))
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            st.error(f"Publish error: {str(e)}")
            return False
    return False

def trigger_prediction():
    """Kirim command untuk trigger predict di Arduino"""
    message = {
        'command': 'predict',
        'timestamp': datetime.now().isoformat(),
        'source': 'dashboard'
    }
    
    if publish_mqtt(MQTT_TOPIC_PUBLISH, message):
        st.success("✅ Prediction request sent to Arduino!")
        return True
    else:
        st.error("❌ Failed to send prediction request")
        return False

def send_control_command(command, value=None):
    """Kirim command kontrol ke Arduino"""
    message = {
        'command': command,
        'value': value,
        'timestamp': datetime.now().isoformat()
    }
    
    if publish_mqtt(MQTT_TOPIC_CONTROL, message):
        st.success(f"✅ Command '{command}' sent")
        return True
    else:
        st.error(f"❌ Failed to send command '{command}'")
        return False

# ==================== LOAD MODELS ====================
def load_all_models():
    """Load semua model .pkl dari folder models"""
    
    if not os.path.exists(MODELS_DIR):
        st.error(f"❌ Folder 'models' tidak ditemukan!")
        st.info(f"Path: {MODELS_DIR}")
        return False
    
    pkl_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    
    if not pkl_files:
        st.warning("📭 No .pkl files found")
        return False
    
    # Reset
    st.session_state.ml_models = {}
    st.session_state.scaler = None
    
    ml_count = 0
    
    for pkl_file in pkl_files:
        try:
            file_path = os.path.join(MODELS_DIR, pkl_file)
            
            # Load file
            try:
                loaded_obj = joblib.load(file_path)
            except:
                with open(file_path, 'rb') as f:
                    loaded_obj = pickle.load(f)
            
            model_name = pkl_file.replace('.pkl', '')
            
            # Cek apakah model ML
            if hasattr(loaded_obj, 'predict'):
                model_type = "Classifier" if hasattr(loaded_obj, 'predict_proba') else "Regressor"
                
                st.session_state.ml_models[model_name] = {
                    'model': loaded_obj,
                    'type': model_type
                }
                ml_count += 1
                
            elif 'scaler' in model_name.lower():
                st.session_state.scaler = loaded_obj
                
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
    
    return ml_count > 0

# ==================== PREDICTION FUNCTIONS ====================
def make_prediction_local(temperature, humidity):
    """Buat prediksi lokal dengan model yang diload"""
    
    if not st.session_state.ml_models:
        return {}
    
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    
    features = np.array([[temperature, humidity, hour, minute]])
    
    # Scale jika ada scaler
    if st.session_state.scaler is not None:
        try:
            features = st.session_state.scaler.transform(features)
        except:
            pass
    
    predictions = {}
    
    for model_name, model_info in st.session_state.ml_models.items():
        model = model_info['model']
        
        try:
            pred = model.predict(features)
            pred_value = pred[0] if isinstance(pred, (list, np.ndarray)) else pred
            
            # Confidence
            confidence = 1.0
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(features)[0]
                    confidence = max(proba)
                except:
                    pass
            
            # Label
            if isinstance(pred_value, (int, np.integer)):
                labels = {0: 'DINGIN', 1: 'NORMAL', 2: 'PANAS'}
                label = labels.get(int(pred_value), f"CLASS_{pred_value}")
            else:
                label = str(pred_value)
            
            predictions[model_name] = {
                'prediction': pred_value,
                'label': label,
                'confidence': float(confidence),
                'model_type': model_info['type']
            }
            
        except Exception as e:
            predictions[model_name] = {
                'prediction': None,
                'label': f"ERROR",
                'confidence': 0.0,
                'model_type': model_info['type'],
                'error': str(e)[:50]
            }
    
    # Save to history
    history_entry = {
        'timestamp': datetime.now(),
        'temperature': temperature,
        'humidity': humidity,
        'predictions': predictions,
        'type': 'local'
    }
    st.session_state.predictions.append(history_entry)
    
    if len(st.session_state.predictions) > 50:
        st.session_state.predictions = st.session_state.predictions[-50:]
    
    return predictions

# ==================== SIDEBAR ====================
def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
        
        # MQTT Connection Section
        st.markdown("---")
        st.subheader("📡 MQTT Connection")
        
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.mqtt_connected:
                if st.button("🔗 Connect MQTT", use_container_width=True, type="primary"):
                    with st.spinner("Connecting..."):
                        if setup_mqtt():
                            st.success("Connected!")
                            time.sleep(1)
                            st.rerun()
        with col2:
            if st.session_state.mqtt_connected:
                if st.button("🔌 Disconnect", use_container_width=True):
                    if st.session_state.mqtt_client:
                        st.session_state.mqtt_client.loop_stop()
                    st.session_state.mqtt_connected = False
                    st.rerun()
        
        # MQTT Status
        if st.session_state.mqtt_connected:
            st.markdown('<p class="status-connected">✅ Connected to HiveMQ</p>', unsafe_allow_html=True)
            st.write(f"**Broker:** {MQTT_BROKER}")
            st.write(f"**Listening:** `{MQTT_TOPIC_SUBSCRIBE}`")
        else:
            st.markdown('<p class="status-disconnected">❌ MQTT Disconnected</p>', unsafe_allow_html=True)
        
        # Model Management
        st.markdown("---")
        st.subheader("🤖 Model Management")
        
        if st.button("🔄 Load ML Models", use_container_width=True):
            with st.spinner("Loading models..."):
                if load_all_models():
                    st.success(f"✅ Loaded {len(st.session_state.ml_models)} models")
                    st.rerun()
                else:
                    st.error("❌ No models loaded")
        
        if st.session_state.ml_models:
            st.write(f"**Models Loaded:** {len(st.session_state.ml_models)}")
            for name in st.session_state.ml_models.keys():
                st.caption(f"• {name}")
        
        # MQTT Control Section
        st.markdown("---")
        st.subheader("🚀 MQTT Controls")
        
        # Trigger Prediction Button
        if st.button("📡 Request Sensor Data", use_container_width=True, type="primary"):
            if st.session_state.mqtt_connected:
                trigger_prediction()
            else:
                st.error("Connect MQTT first!")
        
        # Manual Controls
        st.write("**Device Control:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💡 LED ON", use_container_width=True):
                send_control_command('led', 1)
        with col2:
            if st.button("💡 LED OFF", use_container_width=True):
                send_control_command('led', 0)
        
        if st.button("🔄 System Check", use_container_width=True):
            send_control_command('check', 1)
        
        # Local Prediction
        st.markdown("---")
        st.subheader("🎯 Local Prediction")
        
        temp = st.slider("Temperature (°C)", 15.0, 35.0, 25.0, 0.5)
        hum = st.slider("Humidity (%)", 30.0, 90.0, 65.0, 1.0)
        
        if st.button("🧠 Predict Locally", use_container_width=True):
            make_prediction_local(temp, hum)
            st.rerun()
        
        # Stats
        st.markdown("---")
        st.subheader("📊 Statistics")
        st.write(f"**Sensor Data:** {len(st.session_state.sensor_data)}")
        st.write(f"**Predictions:** {len(st.session_state.predictions)}")
        st.write(f"**MQTT Msgs:** {len(st.session_state.sensor_data)}")
        
        if st.session_state.last_mqtt_msg:
            last_time = st.session_state.last_mqtt_msg['timestamp'].strftime('%H:%M:%S')
            st.write(f"**Last Msg:** {last_time}")

# ==================== MAIN DASHBOARD ====================
def main():
    # Header
    st.markdown("<h1 class='main-title'>🤖 IoT ML Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #666;'>Streamlit Cloud Deployment - No Plotly</h4>", unsafe_allow_html=True)
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
            st.error("📡 MQTT: Disconnected")
    
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
    st.subheader("📈 Sensor Data History")
    
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
        st.info("📭 No sensor data yet. Connect MQTT and request data.")
    
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
                    if 'DINGIN' in label or 'COLD' in label:
                        color = '#3498DB'
                    elif 'PANAS' in label or 'HOT' in label:
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
        st.subheader("📨 Latest MQTT Message")
        
        with st.expander("View Message Details"):
            msg = st.session_state.last_mqtt_msg
            st.write(f"**Topic:** {msg['topic']}")
            st.write(f"**Time:** {msg['timestamp'].strftime('%H:%M:%S')}")
            st.json(msg['data'])
    
    # Row 5: Quick Actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📡 Request Data"):
            if st.session_state.mqtt_connected:
                trigger_prediction()
    
    with col2:
        if st.button("🧠 Predict Latest"):
            if st.session_state.sensor_data:
                latest = st.session_state.sensor_data[-1]
                make_prediction_local(latest['temperature'], latest['humidity'])
                st.rerun()
    
    with col3:
        if st.button("📥 Export Data"):
            if st.session_state.sensor_data:
                sensor_df = pd.DataFrame(st.session_state.sensor_data)
                csv = sensor_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    with col4:
        if st.button("🗑️ Clear Data"):
            st.session_state.sensor_data = []
            st.session_state.predictions = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
        <p><strong>🚀 Deployed on Streamlit Cloud</strong></p>
        <p><strong>MQTT Topics:</strong> 
        Publish: <code>{MQTT_TOPIC_PUBLISH}</code> | 
        Subscribe: <code>{MQTT_TOPIC_SUBSCRIBE}</code>
        </p>
        <p>🕐 Last update: {datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh jika MQTT connected
    if st.session_state.mqtt_connected:
        time.sleep(5)
        st.rerun()

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
