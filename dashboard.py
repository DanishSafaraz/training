import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import time
import warnings
import threading
import queue
import paho.mqtt.client as mqtt
import ssl
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# MQTT Configuration for HiveMQ Cloud
MQTT_BROKER = "48be83e63863499c87afce855025c93e.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764992629489"
MQTT_PASSWORD = ".09yUhd13*nZF?A#rjKT"
MQTT_CLIENT_ID = f"streamlit-iot-ml-{int(time.time())}"

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
    .mqtt-message {
        background: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #1E88E5;
        font-family: monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MQTT CLIENT ====================
class MQTTClient:
    def __init__(self):
        self.client = None
        self.connected = False
        self.message_queue = queue.Queue()
        self.connection_status = "disconnected"
        self.last_message_time = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            self.connection_status = "connected"
            self.reconnect_attempts = 0
            st.session_state.mqtt_connected = True
            st.toast("✅ Successfully connected to HiveMQ Cloud!", icon="✅")
            
            # Subscribe to topics
            client.subscribe(MQTT_TOPIC_SUBSCRIBE)
            client.subscribe(MQTT_TOPIC_CONTROL)
            print(f"Subscribed to topics: {MQTT_TOPIC_SUBSCRIBE}, {MQTT_TOPIC_CONTROL}")
        else:
            self.connected = False
            st.session_state.mqtt_connected = False
            error_messages = {
                1: "Incorrect protocol version",
                2: "Invalid client identifier",
                3: "Server unavailable",
                4: "Bad username or password",
                5: "Not authorized"
            }
            error_msg = error_messages.get(rc, f"Connection failed with code {rc}")
            st.toast(f"❌ Connection failed: {error_msg}", icon="❌")
    
    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        self.connection_status = "disconnected"
        st.session_state.mqtt_connected = False
        if rc != 0:
            print(f"Unexpected disconnection. Reason: {rc}")
            st.toast("⚠️ Disconnected from MQTT broker", icon="⚠️")
            
            # Try to reconnect
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                time.sleep(2 ** self.reconnect_attempts)  # Exponential backoff
                try:
                    client.reconnect()
                except:
                    pass
    
    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            message_data = {
                'timestamp': datetime.now(),
                'topic': msg.topic,
                'payload': payload,
                'qos': msg.qos,
                'retain': msg.retain
            }
            
            # Try to parse as JSON
            try:
                message_data['data'] = json.loads(payload)
            except:
                message_data['data'] = payload
            
            # Put message in queue
            self.message_queue.put(message_data)
            
            # Update last message time
            self.last_message_time = datetime.now()
            
            # Process sensor data
            if msg.topic == MQTT_TOPIC_SUBSCRIBE:
                self.process_sensor_data(message_data)
                
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def process_sensor_data(self, message_data):
        """Process incoming sensor data and store in session state"""
        try:
            if 'data' in message_data and isinstance(message_data['data'], dict):
                data = message_data['data']
                
                # Extract temperature and humidity
                temp = data.get('temperature', data.get('temp', None))
                humid = data.get('humidity', data.get('humid', None))
                
                if temp is not None and humid is not None:
                    sensor_entry = {
                        'timestamp': message_data['timestamp'],
                        'temperature': float(temp),
                        'humidity': float(humid),
                        'source': 'mqtt',
                        'device_id': data.get('device_id', 'unknown'),
                        'raw_data': data
                    }
                    
                    # Add to sensor data history
                    st.session_state.sensor_data.append(sensor_entry)
                    
                    # Keep only last 100 entries
                    if len(st.session_state.sensor_data) > 100:
                        st.session_state.sensor_data = st.session_state.sensor_data[-100:]
                    
                    # Store last message
                    st.session_state.last_mqtt_msg = message_data
                    
                    # Auto-predict if enabled
                    if st.session_state.get('auto_predict', False):
                        self.trigger_prediction(temp, humid)
        
        except Exception as e:
            print(f"Error processing sensor data: {e}")
    
    def trigger_prediction(self, temperature, humidity):
        """Trigger prediction based on sensor data"""
        try:
            if st.session_state.ml_models:
                predictions = make_prediction_local(float(temperature), float(humidity))
                
                # Publish predictions to MQTT
                prediction_msg = {
                    'timestamp': datetime.now().isoformat(),
                    'temperature': temperature,
                    'humidity': humidity,
                    'predictions': predictions
                }
                
                self.publish(MQTT_TOPIC_PUBLISH, json.dumps(prediction_msg))
        except Exception as e:
            print(f"Error in auto-prediction: {e}")
    
    def connect(self):
        """Connect to MQTT broker"""
        try:
            if self.client and self.connected:
                return True
            
            # Create new client
            self.client = mqtt.Client(
                client_id=MQTT_CLIENT_ID,
                clean_session=True,
                protocol=mqtt.MQTTv311
            )
            
            # Set username and password
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            
            # Configure TLS
            self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
            self.client.tls_insecure_set(False)
            
            # Set callbacks
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message
            
            # Connect
            self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            
            # Start network loop in background thread
            self.client.loop_start()
            
            # Wait for connection
            for _ in range(10):  # Wait up to 10 seconds
                if self.connected:
                    return True
                time.sleep(1)
            
            return False
            
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
                self.connected = False
                self.connection_status = "disconnected"
                st.session_state.mqtt_connected = False
        except:
            pass
    
    def publish(self, topic, message, qos=1, retain=False):
        """Publish message to MQTT topic"""
        if self.client and self.connected:
            try:
                result = self.client.publish(topic, message, qos=qos, retain=retain)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    return True
            except Exception as e:
                print(f"Publish error: {e}")
        return False
    
    def get_messages(self):
        """Get all messages from queue"""
        messages = []
        while not self.message_queue.empty():
            try:
                messages.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        return messages

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
if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = MQTTClient()
if 'auto_predict' not in st.session_state:
    st.session_state.auto_predict = False
if 'mqtt_messages' not in st.session_state:
    st.session_state.mqtt_messages = []

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
        'type': 'ml_prediction'
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
            'source': 'demo',
            'device_id': 'demo_device'
        })

# ==================== SIDEBAR ====================
def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
        
        # MQTT Connection Control
        st.markdown("---")
        st.subheader("📡 MQTT Connection")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔗 Connect", use_container_width=True, type="primary"):
                with st.spinner("Connecting to HiveMQ Cloud..."):
                    if st.session_state.mqtt_client.connect():
                        st.success("Connected successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to connect")
        
        with col2:
            if st.button("🔌 Disconnect", use_container_width=True):
                st.session_state.mqtt_client.disconnect()
                st.rerun()
        
        # Connection Status
        if st.session_state.mqtt_connected:
            st.markdown('<p class="status-connected">✅ Connected to HiveMQ Cloud</p>', unsafe_allow_html=True)
            st.caption(f"Client ID: {MQTT_CLIENT_ID[:20]}...")
        else:
            st.markdown('<p class="status-disconnected">❌ MQTT Disconnected</p>', unsafe_allow_html=True)
        
        # MQTT Topics Info
        with st.expander("📡 MQTT Topics"):
            st.write(f"**Subscribe:** `{MQTT_TOPIC_SUBSCRIBE}`")
            st.write(f"**Publish Predictions:** `{MQTT_TOPIC_PUBLISH}`")
            st.write(f"**Control:** `{MQTT_TOPIC_CONTROL}`")
            st.write(f"**Broker:** {MQTT_BROKER}")
        
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
            for name, info in st.session_state.ml_models.items():
                st.caption(f"• {name} ({info['type']}) - Acc: {info['accuracy']:.0%}")
        
        # Auto-Prediction Toggle
        st.markdown("---")
        st.subheader("⚡ Auto Features")
        
        auto_predict = st.toggle("🤖 Auto Predict", 
                                value=st.session_state.auto_predict,
                                help="Automatically make predictions when new sensor data arrives")
        if auto_predict != st.session_state.auto_predict:
            st.session_state.auto_predict = auto_predict
            st.rerun()
        
        # Demo Controls
        st.markdown("---")
        st.subheader("🎯 Manual Controls")
        
        temp = st.slider("Temperature (°C)", 15.0, 35.0, 25.0, 0.5)
        hum = st.slider("Humidity (%)", 30.0, 90.0, 65.0, 1.0)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 Predict", use_container_width=True):
                predictions = make_prediction_local(temp, hum)
                
                # Publish prediction to MQTT
                if st.session_state.mqtt_connected:
                    prediction_msg = {
                        'timestamp': datetime.now().isoformat(),
                        'temperature': temp,
                        'humidity': hum,
                        'predictions': predictions,
                        'source': 'manual'
                    }
                    st.session_state.mqtt_client.publish(
                        MQTT_TOPIC_PUBLISH, 
                        json.dumps(prediction_msg)
                    )
                    st.toast("Prediction published to MQTT!", icon="📤")
                
                st.rerun()
        
        with col2:
            if st.button("📊 Generate Data", use_container_width=True):
                generate_sample_sensor_data()
                st.rerun()
        
        # Publish Test Message
        if st.button("📤 Publish Test Msg", use_container_width=True):
            if st.session_state.mqtt_connected:
                test_msg = {
                    'timestamp': datetime.now().isoformat(),
                    'temperature': round(np.random.uniform(20, 30), 2),
                    'humidity': round(np.random.uniform(40, 80), 2),
                    'device_id': 'streamlit_test',
                    'message': 'Test message from Streamlit'
                }
                
                if st.session_state.mqtt_client.publish(
                    MQTT_TOPIC_SUBSCRIBE, 
                    json.dumps(test_msg)
                ):
                    st.toast("Test message published!", icon="✅")
                else:
                    st.toast("Failed to publish message", icon="❌")
            else:
                st.warning("Connect to MQTT first")
        
        # Stats
        st.markdown("---")
        st.subheader("📊 Statistics")
        st.write(f"**Sensor Data:** {len(st.session_state.sensor_data)}")
        st.write(f"**Predictions:** {len(st.session_state.predictions)}")
        
        if st.session_state.last_mqtt_msg:
            last_time = st.session_state.last_mqtt_msg['timestamp'].strftime('%H:%M:%S')
            st.write(f"**Last Msg:** {last_time}")
        
        # Clear Data
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.sensor_data = []
            st.session_state.predictions = []
            st.session_state.mqtt_messages = []
            st.rerun()

# ==================== MQTT MESSAGES PANEL ====================
def render_mqtt_messages():
    """Render MQTT messages panel"""
    st.subheader("📨 MQTT Messages")
    
    # Process incoming messages
    if st.session_state.mqtt_connected:
        messages = st.session_state.mqtt_client.get_messages()
        for msg in messages:
            st.session_state.mqtt_messages.append(msg)
            
            # Keep only last 20 messages
            if len(st.session_state.mqtt_messages) > 20:
                st.session_state.mqtt_messages = st.session_state.mqtt_messages[-20:]
    
    # Display messages
    if st.session_state.mqtt_messages:
        messages_df = pd.DataFrame([
            {
                'Time': msg['timestamp'].strftime('%H:%M:%S'),
                'Topic': msg['topic'],
                'Payload': str(msg['data'])[:100] + '...' if len(str(msg['data'])) > 100 else str(msg['data'])
            }
            for msg in reversed(st.session_state.mqtt_messages[-10:])
        ])
        
        st.dataframe(
            messages_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Time': st.column_config.TextColumn(width="small"),
                'Topic': st.column_config.TextColumn(width="medium"),
                'Payload': st.column_config.TextColumn(width="large")
            }
        )
        
        # Message details expander
        with st.expander("📋 View Message Details"):
            for msg in reversed(st.session_state.mqtt_messages[-5:]):
                st.markdown(f"""
                <div class="mqtt-message">
                    <strong>Time:</strong> {msg['timestamp'].strftime('%H:%M:%S.%f')[:-3]}<br>
                    <strong>Topic:</strong> {msg['topic']}<br>
                    <strong>QoS:</strong> {msg['qos']} | <strong>Retain:</strong> {msg['retain']}<br>
                    <strong>Data:</strong>
                </div>
                """, unsafe_allow_html=True)
                st.json(msg['data'], expanded=False)
                st.markdown("---")
    else:
        st.info("No MQTT messages received yet. Connect and start receiving data.")

# ==================== MAIN DASHBOARD ====================
def main():
    # Header
    st.markdown("<h1 class='main-title'>🤖 IoT ML Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #666;'>Real-time MQTT + Machine Learning Integration</h4>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Render sidebar
    render_sidebar()
    
    # Row 1: Status Cards
    st.subheader("📊 Dashboard Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.mqtt_connected:
            st.success("📡 MQTT: Connected")
            st.caption(f"To: {MQTT_BROKER}")
        else:
            st.info("📡 MQTT: Disconnected")
    
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
    
    # Row 2: MQTT Messages
    render_mqtt_messages()
    
    st.markdown("---")
    
    # Row 3: Sensor Data
    st.subheader("📈 Sensor Data")
    
    if st.session_state.sensor_data:
        sensor_df = pd.DataFrame(st.session_state.sensor_data)
        
        # Tampilkan data dalam tabel
        st.dataframe(
            sensor_df.tail(10).sort_values('timestamp', ascending=False),
            use_container_width=True,
            column_config={
                'timestamp': st.column_config.DatetimeColumn(
                    label="Time", 
                    format="HH:mm:ss",
                    width="small"
                ),
                'temperature': st.column_config.NumberColumn(
                    label="Temp", 
                    format="%.1f °C",
                    width="small"
                ),
                'humidity': st.column_config.NumberColumn(
                    label="Humid", 
                    format="%.1f %",
                    width="small"
                ),
                'source': st.column_config.TextColumn(
                    label="Source",
                    width="small"
                ),
                'device_id': st.column_config.TextColumn(
                    label="Device",
                    width="medium"
                )
            }
        )
        
        # Charts
        if len(sensor_df) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌡️ Temperature Trend")
                chart_data = pd.DataFrame({
                    'Time': sensor_df['timestamp'].dt.strftime('%H:%M'),
                    'Temperature': sensor_df['temperature']
                }).set_index('Time')
                st.line_chart(chart_data)
            
            with col2:
                st.subheader("💧 Humidity Trend")
                chart_data = pd.DataFrame({
                    'Time': sensor_df['timestamp'].dt.strftime('%H:%M'),
                    'Humidity': sensor_df['humidity']
                }).set_index('Time')
                st.line_chart(chart_data)
    else:
        st.info("📭 No sensor data yet. Connect to MQTT or generate sample data.")
    
    st.markdown("---")
    
    # Row 4: Predictions
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
                    
                    # Publish button for each prediction
                    if st.session_state.mqtt_connected:
                        if st.button(f"📤 Publish {model_name}", key=f"publish_{idx}"):
                            pred_msg = {
                                'model': model_name,
                                'prediction': pred_info['label'],
                                'confidence': pred_info['confidence'],
                                'temperature': latest_pred['temperature'],
                                'humidity': latest_pred['humidity'],
                                'timestamp': datetime.now().isoformat()
                            }
                            st.session_state.mqtt_client.publish(
                                MQTT_TOPIC_PUBLISH,
                                json.dumps(pred_msg)
                            )
                            st.toast(f"Published {model_name} prediction!", icon="📤")
        
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
    
    # Row 5: Control Panel
    st.markdown("---")
    st.subheader("🎛️ Control Panel")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.session_state.sensor_data:
            if st.button("🧠 Predict Latest", use_container_width=True):
                latest = st.session_state.sensor_data[-1]
                make_prediction_local(latest['temperature'], latest['humidity'])
                st.rerun()
    
    with col3:
        if st.button("📤 Send Control", use_container_width=True):
            if st.session_state.mqtt_connected:
                control_msg = {
                    'command': 'predict',
                    'timestamp': datetime.now().isoformat(),
                    'source': 'streamlit_dashboard'
                }
                st.session_state.mqtt_client.publish(
                    MQTT_TOPIC_CONTROL,
                    json.dumps(control_msg)
                )
                st.toast("Control command sent!", icon="🎛️")
            else:
                st.warning("Connect to MQTT first")
    
    with col4:
        if st.button("📊 Stats", use_container_width=True):
            st.info(f"""
            **Dashboard Statistics:**
            - MQTT Messages: {len(st.session_state.mqtt_messages)}
            - Sensor Readings: {len(st.session_state.sensor_data)}
            - Predictions: {len(st.session_state.predictions)}
            - Models: {len(st.session_state.ml_models)}
            - Auto Predict: {'On' if st.session_state.auto_predict else 'Off'}
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
        <p><strong>🚀 IoT ML Dashboard - Connected to HiveMQ Cloud</strong></p>
        <p>📡 Broker: <code>{MQTT_BROKER}</code> | 🔌 Status: <strong>{'Connected' if st.session_state.mqtt_connected else 'Disconnected'}</strong></p>
        <p>⚠️ <strong>Note:</strong> Requires paho-mqtt and TLS/SSL support</p>
        <p>🕐 Last update: {datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== RUN APP ====================
if __name__ == "__main__":
    # Check for required packages
    try:
        import paho.mqtt.client
    except ImportError:
        st.error("""
        ❌ **Missing Required Package**
        
        Please install paho-mqtt to use MQTT features:
        ```bash
        pip install paho-mqtt
        ```
        
        The app will run in demo mode without MQTT.
        """)
    
    # Run main app
    main()
    
    # Cleanup on app close
    if st.session_state.mqtt_connected:
        st.session_state.mqtt_client.disconnect()
