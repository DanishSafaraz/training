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
import socket
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# MQTT Configuration for HiveMQ Cloud
MQTT_BROKER = "48be83e63863499c87afce855025c93e.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764992629489"
MQTT_PASSWORD = ".09yUhd13*nZF?A#rjKT"
MQTT_CLIENT_ID = f"streamlit-iot-{int(time.time())}"

# Topics
MQTT_TOPIC_SUBSCRIBE = "iot/sensor/data"
MQTT_TOPIC_PUBLISH = "iot/predict/ml"
MQTT_TOPIC_CONTROL = "iot/control/ml"

# Keepalive interval (seconds)
MQTT_KEEPALIVE = 60
MQTT_CONNECT_TIMEOUT = 10

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
    .status-connecting {
        color: #F39C12;
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
    .connection-stats {
        background: #f0f7ff;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MQTT CLIENT ====================
class MQTTClient:
    def __init__(self):
        self.client = None
        self.connected = False
        self.connecting = False
        self.message_queue = queue.Queue()
        self.connection_status = "disconnected"
        self.last_message_time = None
        self.last_connect_time = None
        self.last_disconnect_time = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.total_messages_received = 0
        self.connection_thread = None
        self.lock = threading.Lock()
        
        # Stats
        self.stats = {
            'connect_success': 0,
            'connect_failed': 0,
            'messages_received': 0,
            'messages_published': 0,
            'last_error': None,
            'last_error_time': None
        }
    
    def on_connect(self, client, userdata, flags, rc):
        with self.lock:
            if rc == 0:
                self.connected = True
                self.connecting = False
                self.connection_status = "connected"
                self.reconnect_attempts = 0
                self.last_connect_time = datetime.now()
                self.stats['connect_success'] += 1
                
                # Subscribe to topics
                try:
                    client.subscribe(MQTT_TOPIC_SUBSCRIBE, qos=1)
                    client.subscribe(MQTT_TOPIC_CONTROL, qos=1)
                    print(f"✅ Subscribed to topics: {MQTT_TOPIC_SUBSCRIBE}, {MQTT_TOPIC_CONTROL}")
                    
                    # Store in session state for UI
                    st.session_state.mqtt_connected = True
                    st.session_state.mqtt_connection_time = self.last_connect_time
                    
                    # Show success message
                    st.toast("✅ Successfully connected to HiveMQ Cloud!", icon="✅")
                    
                except Exception as e:
                    print(f"❌ Subscription error: {e}")
                    self.stats['last_error'] = f"Subscribe error: {e}"
                    self.stats['last_error_time'] = datetime.now()
                    
            else:
                self.connected = False
                self.connecting = False
                self.connection_status = f"failed_{rc}"
                self.stats['connect_failed'] += 1
                self.stats['last_error'] = f"Connection failed with code: {rc}"
                self.stats['last_error_time'] = datetime.now()
                
                error_messages = {
                    1: "Incorrect protocol version",
                    2: "Invalid client identifier",
                    3: "Server unavailable",
                    4: "Bad username or password",
                    5: "Not authorized"
                }
                error_msg = error_messages.get(rc, f"Connection failed with code {rc}")
                
                st.session_state.mqtt_connected = False
                st.toast(f"❌ Connection failed: {error_msg}", icon="❌")
    
    def on_disconnect(self, client, userdata, rc):
        with self.lock:
            self.connected = False
            self.connecting = False
            self.connection_status = "disconnected"
            self.last_disconnect_time = datetime.now()
            
            st.session_state.mqtt_connected = False
            
            if rc != 0:
                print(f"⚠️ Unexpected disconnection. Reason: {rc}")
                self.stats['last_error'] = f"Unexpected disconnect: {rc}"
                self.stats['last_error_time'] = datetime.now()
                
                # Don't auto-reconnect in Streamlit - let user control it
                st.toast("⚠️ Disconnected from MQTT broker", icon="⚠️")
            else:
                print("ℹ️ Normal disconnection")
    
    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            message_data = {
                'timestamp': datetime.now(),
                'topic': msg.topic,
                'payload': payload,
                'qos': msg.qos,
                'retain': msg.retain,
                'message_id': msg.mid
            }
            
            # Try to parse as JSON
            try:
                message_data['data'] = json.loads(payload)
            except json.JSONDecodeError:
                message_data['data'] = payload
            
            # Put message in queue
            self.message_queue.put(message_data)
            
            # Update stats
            with self.lock:
                self.last_message_time = datetime.now()
                self.total_messages_received += 1
                self.stats['messages_received'] += 1
            
            # Process based on topic
            if msg.topic == MQTT_TOPIC_SUBSCRIBE:
                self.process_sensor_data(message_data)
                
            print(f"📨 Received message on {msg.topic}: {payload[:100]}...")
                
        except Exception as e:
            print(f"❌ Error processing message: {e}")
            with self.lock:
                self.stats['last_error'] = f"Message processing error: {e}"
                self.stats['last_error_time'] = datetime.now()
    
    def process_sensor_data(self, message_data):
        """Process incoming sensor data and store in session state"""
        try:
            if 'data' in message_data and isinstance(message_data['data'], dict):
                data = message_data['data']
                
                # Extract temperature and humidity
                temp = data.get('temperature', data.get('temp', None))
                humid = data.get('humidity', data.get('humid', None))
                
                if temp is not None:
                    try:
                        temp = float(temp)
                    except:
                        temp = None
                
                if humid is not None:
                    try:
                        humid = float(humid)
                    except:
                        humid = None
                
                if temp is not None and humid is not None:
                    sensor_entry = {
                        'timestamp': message_data['timestamp'],
                        'temperature': temp,
                        'humidity': humid,
                        'source': 'mqtt',
                        'device_id': data.get('device_id', 'unknown'),
                        'topic': message_data['topic'],
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
            print(f"❌ Error processing sensor data: {e}")
            with self.lock:
                self.stats['last_error'] = f"Sensor data error: {e}"
                self.stats['last_error_time'] = datetime.now()
    
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
                    'predictions': predictions,
                    'source': 'auto_predict'
                }
                
                self.publish(MQTT_TOPIC_PUBLISH, json.dumps(prediction_msg))
        except Exception as e:
            print(f"❌ Error in auto-prediction: {e}")
    
    def connect_async(self):
        """Connect to MQTT broker in a separate thread"""
        if self.connecting or self.connected:
            return False
        
        self.connecting = True
        self.connection_status = "connecting"
        
        def connection_thread():
            try:
                # Create new client
                self.client = mqtt.Client(
                    client_id=MQTT_CLIENT_ID,
                    clean_session=True,
                    protocol=mqtt.MQTTv311,
                    transport="tcp"
                )
                
                # Set username and password
                self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
                
                # Configure TLS
                self.client.tls_set(
                    cert_reqs=ssl.CERT_REQUIRED,
                    tls_version=ssl.PROTOCOL_TLS
                )
                self.client.tls_insecure_set(False)
                
                # Set callbacks
                self.client.on_connect = self.on_connect
                self.client.on_disconnect = self.on_disconnect
                self.client.on_message = self.on_message
                
                # Set last will message
                self.client.will_set(
                    MQTT_TOPIC_CONTROL,
                    payload=json.dumps({
                        'client_id': MQTT_CLIENT_ID,
                        'status': 'offline',
                        'timestamp': datetime.now().isoformat()
                    }),
                    qos=1,
                    retain=False
                )
                
                print(f"🔗 Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
                
                # Connect with timeout
                self.client.connect_async(
                    MQTT_BROKER, 
                    MQTT_PORT, 
                    keepalive=MQTT_KEEPALIVE
                )
                
                # Start network loop
                self.client.loop_start()
                
                # Wait for connection with timeout
                timeout = MQTT_CONNECT_TIMEOUT
                while timeout > 0 and not self.connected and self.connecting:
                    time.sleep(0.5)
                    timeout -= 0.5
                
                if not self.connected:
                    print(f"❌ Connection timeout after {MQTT_CONNECT_TIMEOUT} seconds")
                    with self.lock:
                        self.connecting = False
                        self.connection_status = "timeout"
                        self.stats['connect_failed'] += 1
                        self.stats['last_error'] = "Connection timeout"
                        self.stats['last_error_time'] = datetime.now()
                    
                    # Clean up
                    if self.client:
                        self.client.loop_stop()
                        self.client.disconnect()
                    
                    st.toast("❌ Connection timeout", icon="❌")
                
            except socket.gaierror as e:
                print(f"❌ DNS resolution error: {e}")
                with self.lock:
                    self.connecting = False
                    self.connection_status = "dns_error"
                    self.stats['connect_failed'] += 1
                    self.stats['last_error'] = f"DNS error: {e}"
                    self.stats['last_error_time'] = datetime.now()
                st.toast(f"❌ DNS error: {e}", icon="❌")
                
            except ssl.SSLError as e:
                print(f"❌ SSL error: {e}")
                with self.lock:
                    self.connecting = False
                    self.connection_status = "ssl_error"
                    self.stats['connect_failed'] += 1
                    self.stats['last_error'] = f"SSL error: {e}"
                    self.stats['last_error_time'] = datetime.now()
                st.toast(f"❌ SSL error: {e}", icon="❌")
                
            except Exception as e:
                print(f"❌ Connection error: {e}")
                with self.lock:
                    self.connecting = False
                    self.connection_status = "error"
                    self.stats['connect_failed'] += 1
                    self.stats['last_error'] = f"Connection error: {e}"
                    self.stats['last_error_time'] = datetime.now()
                st.toast(f"❌ Connection error: {e}", icon="❌")
        
        # Start connection thread
        self.connection_thread = threading.Thread(target=connection_thread, daemon=True)
        self.connection_thread.start()
        
        return True
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        with self.lock:
            self.connecting = False
            
            if self.connected and self.client:
                try:
                    # Send disconnect message
                    disconnect_msg = {
                        'client_id': MQTT_CLIENT_ID,
                        'status': 'disconnecting',
                        'timestamp': datetime.now().isoformat()
                    }
                    self.client.publish(
                        MQTT_TOPIC_CONTROL,
                        json.dumps(disconnect_msg),
                        qos=1
                    )
                    
                    # Disconnect
                    self.client.loop_stop()
                    self.client.disconnect()
                    
                    self.connected = False
                    self.connection_status = "disconnected"
                    st.session_state.mqtt_connected = False
                    
                    print("🔌 Disconnected from MQTT broker")
                    st.toast("🔌 Disconnected from MQTT", icon="🔌")
                    
                except Exception as e:
                    print(f"❌ Error during disconnect: {e}")
            else:
                self.connected = False
                self.connection_status = "disconnected"
                st.session_state.mqtt_connected = False
    
    def publish(self, topic, message, qos=1, retain=False):
        """Publish message to MQTT topic"""
        if self.client and self.connected:
            try:
                result = self.client.publish(topic, message, qos=qos, retain=retain)
                
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    with self.lock:
                        self.stats['messages_published'] += 1
                    return True
                else:
                    print(f"❌ Publish failed with code: {result.rc}")
                    with self.lock:
                        self.stats['last_error'] = f"Publish failed: {result.rc}"
                        self.stats['last_error_time'] = datetime.now()
            except Exception as e:
                print(f"❌ Publish error: {e}")
                with self.lock:
                    self.stats['last_error'] = f"Publish error: {e}"
                    self.stats['last_error_time'] = datetime.now()
        else:
            print("❌ Cannot publish: Not connected")
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
    
    def get_connection_stats(self):
        """Get connection statistics"""
        with self.lock:
            stats = self.stats.copy()
            stats['connected'] = self.connected
            stats['connecting'] = self.connecting
            stats['connection_status'] = self.connection_status
            stats['last_connect_time'] = self.last_connect_time
            stats['last_disconnect_time'] = self.last_disconnect_time
            stats['last_message_time'] = self.last_message_time
            stats['total_messages'] = self.total_messages_received
            return stats

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
if 'mqtt_connection_time' not in st.session_state:
    st.session_state.mqtt_connection_time = None

# ==================== SIMULATED FUNCTIONS ====================
def load_all_models():
    """Simulasi load models"""
    st.session_state.ml_models = {
        'Decision Tree': {'type': 'Classifier', 'accuracy': 0.85},
        'KNN': {'type': 'Classifier', 'accuracy': 0.82},
        'Logistic Regression': {'type': 'Classifier', 'accuracy': 0.80}
    }
    return True

def make_prediction_local(temperature, humidity):
    """Simulasi prediksi"""
    if not st.session_state.ml_models:
        return {}
    
    predictions = {}
    
    for model_name, model_info in st.session_state.ml_models.items():
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
        
        client = st.session_state.mqtt_client
        stats = client.get_connection_stats()
        
        # Connection Status
        if client.connected:
            st.markdown('<p class="status-connected">✅ Connected to HiveMQ</p>', unsafe_allow_html=True)
            if st.session_state.mqtt_connection_time:
                connect_time = st.session_state.mqtt_connection_time.strftime('%H:%M:%S')
                st.caption(f"Connected at: {connect_time}")
        elif client.connecting:
            st.markdown('<p class="status-connecting">🔄 Connecting...</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-disconnected">❌ Disconnected</p>', unsafe_allow_html=True)
        
        # Connection Buttons
        col1, col2 = st.columns(2)
        with col1:
            if not client.connected and not client.connecting:
                if st.button("🔗 Connect", use_container_width=True, type="primary"):
                    with st.spinner("Connecting to HiveMQ Cloud..."):
                        client.connect_async()
                        st.rerun()
        
        with col2:
            if client.connected or client.connecting:
                if st.button("🔌 Disconnect", use_container_width=True):
                    client.disconnect()
                    st.rerun()
        
        # Connection Stats
        with st.expander("📊 Connection Stats"):
            st.write(f"**Status:** {stats['connection_status']}")
            st.write(f"**Messages Received:** {stats['messages_received']}")
            st.write(f"**Messages Published:** {stats['messages_published']}")
            st.write(f"**Connects Success:** {stats['connect_success']}")
            st.write(f"**Connects Failed:** {stats['connect_failed']}")
            
            if stats['last_error'] and stats['last_error_time']:
                error_time = stats['last_error_time'].strftime('%H:%M:%S')
                st.write(f"**Last Error:** {stats['last_error']}")
                st.caption(f"at {error_time}")
        
        # MQTT Topics Info
        with st.expander("📡 MQTT Topics"):
            st.code(f"""
Subscribe:    {MQTT_TOPIC_SUBSCRIBE}
Publish:      {MQTT_TOPIC_PUBLISH}
Control:      {MQTT_TOPIC_CONTROL}
Broker:       {MQTT_BROKER}:{MQTT_PORT}
Client ID:    {MQTT_CLIENT_ID}
            """)
        
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
        
        # Test Connection
        st.markdown("---")
        st.subheader("🧪 Test Connection")
        
        if st.button("🔄 Test MQTT Connection", use_container_width=True):
            if client.connected:
                test_msg = {
                    'test': 'connection_test',
                    'timestamp': datetime.now().isoformat(),
                    'client_id': MQTT_CLIENT_ID,
                    'message': 'Test message from Streamlit Dashboard'
                }
                
                if client.publish(MQTT_TOPIC_CONTROL, json.dumps(test_msg)):
                    st.toast("✅ Test message sent!", icon="✅")
                else:
                    st.toast("❌ Failed to send test message", icon="❌")
            else:
                st.warning("Not connected to MQTT")
        
        # Send Test Sensor Data
        if st.button("📤 Send Test Sensor Data", use_container_width=True):
            if client.connected:
                sensor_data = {
                    'temperature': round(np.random.uniform(20, 30), 2),
                    'humidity': round(np.random.uniform(40, 80), 2),
                    'device_id': 'streamlit_test',
                    'timestamp': datetime.now().isoformat(),
                    'test': True
                }
                
                if client.publish(MQTT_TOPIC_SUBSCRIBE, json.dumps(sensor_data)):
                    st.toast("✅ Test sensor data sent!", icon="✅")
                else:
                    st.toast("❌ Failed to send sensor data", icon="❌")
            else:
                st.warning("Not connected to MQTT")
        
        # Stats
        st.markdown("---")
        st.subheader("📊 Dashboard Stats")
        st.write(f"**Sensor Data:** {len(st.session_state.sensor_data)}")
        st.write(f"**Predictions:** {len(st.session_state.predictions)}")
        st.write(f"**MQTT Messages:** {len(st.session_state.mqtt_messages)}")
        
        if st.session_state.last_mqtt_msg:
            last_time = st.session_state.last_mqtt_msg['timestamp'].strftime('%H:%M:%S')
            st.write(f"**Last Msg:** {last_time}")
        
        # Clear Data
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.sensor_data = []
            st.session_state.predictions = []
            st.session_state.mqtt_messages = []
            st.rerun()

# ==================== MAIN DASHBOARD ====================
def main():
    # Header
    st.markdown("<h1 class='main-title'>🤖 IoT ML Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #666;'>Real-time MQTT + Machine Learning Integration</h4>", unsafe_allow_html=True)
    
    # Quick Status Bar
    client = st.session_state.mqtt_client
    stats = client.get_connection_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if client.connected:
            st.success("✅ Connected")
            if st.session_state.mqtt_connection_time:
                uptime = datetime.now() - st.session_state.mqtt_connection_time
                st.caption(f"Uptime: {str(uptime).split('.')[0]}")
        elif client.connecting:
            st.warning("🔄 Connecting...")
        else:
            st.error("❌ Disconnected")
    
    with col2:
        st.metric("📨 Messages", stats['messages_received'])
    
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
    
    # Render sidebar
    render_sidebar()
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["📡 MQTT Messages", "📈 Sensor Data", "🔮 Predictions"])
    
    with tab1:
        # MQTT Messages
        st.subheader("📨 Live MQTT Messages")
        
        # Process incoming messages
        if client.connected:
            messages = client.get_messages()
            for msg in messages:
                st.session_state.mqtt_messages.append(msg)
                
                # Keep only last 50 messages
                if len(st.session_state.mqtt_messages) > 50:
                    st.session_state.mqtt_messages = st.session_state.mqtt_messages[-50:]
        
        # Display messages
        if st.session_state.mqtt_messages:
            # Message counter
            st.caption(f"Showing last {min(10, len(st.session_state.mqtt_messages))} of {len(st.session_state.mqtt_messages)} messages")
            
            # Display recent messages
            for msg in reversed(st.session_state.mqtt_messages[-10:]):
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="connection-stats">
                            <strong>Time:</strong> {msg['timestamp'].strftime('%H:%M:%S')}<br>
                            <strong>Topic:</strong> {msg['topic']}<br>
                            <strong>QoS:</strong> {msg['qos']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.json(msg['data'], expanded=False)
            
            # Detailed view expander
            with st.expander("📋 View All Message Details"):
                messages_df = pd.DataFrame([
                    {
                        'Time': m['timestamp'].strftime('%H:%M:%S.%f')[:-3],
                        'Topic': m['topic'],
                        'QoS': m['qos'],
                        'Retain': m['retain'],
                        'Payload': str(m['data'])[:150] + '...' if len(str(m['data'])) > 150 else str(m['data'])
                    }
                    for m in reversed(st.session_state.mqtt_messages)
                ])
                
                st.dataframe(
                    messages_df,
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No MQTT messages received yet. Connect to MQTT and start receiving data.")
            
            # Show test instructions
            with st.expander("🧪 How to test MQTT connection"):
                st.markdown("""
                1. **Connect** using the button in the sidebar
                2. **Send test data** using the buttons in the sidebar
                3. **Use MQTT client** to publish to topic:
                   ```
                   Topic: iot/sensor/data
                   Message: {"temperature": 25.5, "humidity": 65.2}
                   ```
                4. **Monitor** messages in this panel
                """)
    
    with tab2:
        # Sensor Data
        st.subheader("📈 Sensor Data History")
        
        if st.session_state.sensor_data:
            sensor_df = pd.DataFrame(st.session_state.sensor_data)
            
            # Tampilkan data dalam tabel
            st.dataframe(
                sensor_df.tail(20).sort_values('timestamp', ascending=False),
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
                    chart_data = sensor_df.set_index('timestamp')[['temperature']].tail(50)
                    st.line_chart(chart_data)
                
                with col2:
                    st.subheader("💧 Humidity Trend")
                    chart_data = sensor_df.set_index('timestamp')[['humidity']].tail(50)
                    st.line_chart(chart_data)
        else:
            st.info("No sensor data yet. Connect to MQTT or generate sample data.")
    
    with tab3:
        # Predictions
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
                
                # Prediction History
                with st.expander("📜 View Prediction History"):
                    history_data = []
                    for pred in st.session_state.predictions[-10:]:
                        row = {
                            'Time': pred['timestamp'].strftime('%H:%M:%S'),
                            'Temp': f"{pred['temperature']:.1f}°C",
                            'Humid': f"{pred['humidity']:.1f}%",
                            'Type': pred['type']
                        }
                        
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
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
        <p><strong>🚀 IoT ML Dashboard - HiveMQ Cloud Integration</strong></p>
        <p>📡 Broker: <code>{MQTT_BROKER}</code> | Client ID: <code>{MQTT_CLIENT_ID}</code></p>
        <p>🔄 Auto-refresh: Every 2 seconds | 📊 Messages: {stats['messages_received']}</p>
        <p>🕐 Dashboard time: {datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh
    time.sleep(2)
    st.rerun()

# ==================== RUN APP ====================
if __name__ == "__main__":
    try:
        # Check if paho-mqtt is installed
        import paho.mqtt.client as mqtt
        
        # Run main app
        main()
        
    except ImportError:
        st.error("""
        ## ❌ Missing Required Package
        
        Please install **paho-mqtt** to use MQTT features:
        
        ```bash
        pip install paho-mqtt
        ```
        
        For Streamlit Cloud, add to `requirements.txt`:
        ```txt
        paho-mqtt==1.6.1
        ```
        
        The app will run in demo mode without MQTT.
        """)
        
        # Run in demo mode
        if st.button("🔄 Continue in Demo Mode"):
            main()
