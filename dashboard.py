import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import time
import warnings
import threading
import queue
import paho.mqtt.client as mqtt
import ssl
import socket
import random
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
    .sensor-data-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
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
                    result_subscribe = client.subscribe(MQTT_TOPIC_SUBSCRIBE, qos=1)
                    result_control = client.subscribe(MQTT_TOPIC_CONTROL, qos=1)
                    print(f"✅ Subscribed to topics: {MQTT_TOPIC_SUBSCRIBE}, {MQTT_TOPIC_CONTROL}")
                    print(f"Subscribe result: {result_subscribe}, Control result: {result_control}")
                    
                    # Store in session state for UI
                    st.session_state.mqtt_connected = True
                    st.session_state.mqtt_connection_time = self.last_connect_time
                    
                    # Show success message
                    if 'toast_shown' not in st.session_state:
                        st.session_state.toast_shown = True
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
                if 'toast_shown' not in st.session_state:
                    st.session_state.toast_shown = True
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
                
                if 'toast_shown' not in st.session_state:
                    st.session_state.toast_shown = True
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
                parsed_data = json.loads(payload)
                message_data['data'] = parsed_data
                print(f"✅ Parsed JSON data: {parsed_data}")
            except json.JSONDecodeError:
                message_data['data'] = payload
                print(f"⚠️ Raw payload (not JSON): {payload}")
            
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
                
            print(f"📨 Received message on {msg.topic}")
                
        except Exception as e:
            print(f"❌ Error processing message: {e}")
            with self.lock:
                self.stats['last_error'] = f"Message processing error: {e}"
                self.stats['last_error_time'] = datetime.now()
    
    def process_sensor_data(self, message_data):
        """Process incoming sensor data and store in session state"""
        try:
            print(f"🔧 Processing sensor data: {message_data}")
            
            data_to_store = {}
            
            if 'data' in message_data:
                if isinstance(message_data['data'], dict):
                    data = message_data['data']
                    
                    # Extract temperature - try multiple possible keys
                    temp = None
                    for key in ['temperature', 'temp', 'Temperature', 'Temp', 'TEMPERATURE', 'TEMP']:
                        if key in data:
                            try:
                                temp = float(data[key])
                                break
                            except (ValueError, TypeError):
                                pass
                    
                    # Extract humidity - try multiple possible keys
                    humid = None
                    for key in ['humidity', 'humid', 'Humidity', 'Humid', 'HUMIDITY', 'HUMID']:
                        if key in data:
                            try:
                                humid = float(data[key])
                                break
                            except (ValueError, TypeError):
                                pass
                    
                    # If no temperature/humidity found, check for nested structures
                    if temp is None or humid is None:
                        # Check if data is a string that might contain the values
                        if isinstance(data, str):
                            # Try to extract numbers from string
                            import re
                            numbers = re.findall(r'\d+\.?\d*', data)
                            if len(numbers) >= 2:
                                temp = float(numbers[0]) if temp is None else temp
                                humid = float(numbers[1]) if humid is None else humid
                    
                    print(f"📊 Extracted - Temp: {temp}, Humid: {humid}")
                    
                    if temp is not None and humid is not None:
                        sensor_entry = {
                            'timestamp': message_data['timestamp'],
                            'temperature': temp,
                            'humidity': humid,
                            'source': 'mqtt',
                            'device_id': data.get('device_id', data.get('device', 'unknown')),
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
                        
                        print(f"✅ Stored sensor data: Temp={temp}, Humid={humid}")
                        
                        # Auto-predict if enabled
                        if st.session_state.get('auto_predict', False):
                            self.trigger_prediction(temp, humid)
                    
                    # Also store the raw message for display
                    st.session_state.mqtt_messages.append(message_data)
                    
                    # Keep only last 50 messages
                    if len(st.session_state.mqtt_messages) > 50:
                        st.session_state.mqtt_messages = st.session_state.mqtt_messages[-50:]
            
        except Exception as e:
            print(f"❌ Error processing sensor data: {e}")
            import traceback
            traceback.print_exc()
            with self.lock:
                self.stats['last_error'] = f"Sensor data error: {e}"
                self.stats['last_error_time'] = datetime.now()
    
    def trigger_prediction(self, temperature, humidity):
        """Trigger prediction based on sensor data"""
        try:
            print(f"🤖 Triggering prediction for Temp={temperature}, Humid={humidity}")
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
                print(f"✅ Published prediction to MQTT")
        except Exception as e:
            print(f"❌ Error in auto-prediction: {e}")
            import traceback
            traceback.print_exc()
    
    def connect_async(self):
        """Connect to MQTT broker in a separate thread"""
        if self.connecting or self.connected:
            return False
        
        self.connecting = True
        self.connection_status = "connecting"
        st.session_state.toast_shown = False  # Reset toast flag
        
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
                print(f"📝 Username: {MQTT_USERNAME}")
                print(f"🔑 Password: {'*' * len(MQTT_PASSWORD)}")
                
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
                    
                    if not st.session_state.toast_shown:
                        st.session_state.toast_shown = True
                        st.toast("❌ Connection timeout", icon="❌")
                
            except socket.gaierror as e:
                print(f"❌ DNS resolution error: {e}")
                with self.lock:
                    self.connecting = False
                    self.connection_status = "dns_error"
                    self.stats['connect_failed'] += 1
                    self.stats['last_error'] = f"DNS error: {e}"
                    self.stats['last_error_time'] = datetime.now()
                
                if not st.session_state.toast_shown:
                    st.session_state.toast_shown = True
                    st.toast(f"❌ DNS error: {e}", icon="❌")
                
            except ssl.SSLError as e:
                print(f"❌ SSL error: {e}")
                with self.lock:
                    self.connecting = False
                    self.connection_status = "ssl_error"
                    self.stats['connect_failed'] += 1
                    self.stats['last_error'] = f"SSL error: {e}"
                    self.stats['last_error_time'] = datetime.now()
                
                if not st.session_state.toast_shown:
                    st.session_state.toast_shown = True
                    st.toast(f"❌ SSL error: {e}", icon="❌")
                
            except Exception as e:
                print(f"❌ Connection error: {e}")
                import traceback
                traceback.print_exc()
                with self.lock:
                    self.connecting = False
                    self.connection_status = "error"
                    self.stats['connect_failed'] += 1
                    self.stats['last_error'] = f"Connection error: {e}"
                    self.stats['last_error_time'] = datetime.now()
                
                if not st.session_state.toast_shown:
                    st.session_state.toast_shown = True
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
                    print(f"✅ Published to {topic}: {message[:100]}...")
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
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'ml_models': {},
        'predictions': [],
        'sensor_data': [],
        'mqtt_connected': False,
        'last_mqtt_msg': None,
        'mqtt_client': None,
        'auto_predict': False,
        'mqtt_messages': [],
        'mqtt_connection_time': None,
        'toast_shown': False,
        'demo_data_generated': False,
        'last_prediction_time': None,
        'chart_data': pd.DataFrame(),
        'initialized': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Initialize MQTT client
    if st.session_state.mqtt_client is None:
        st.session_state.mqtt_client = MQTTClient()
    
    # Generate initial demo data if not exists
    if not st.session_state.demo_data_generated:
        generate_initial_demo_data()
        st.session_state.demo_data_generated = True
    
    st.session_state.initialized = True

def generate_initial_demo_data():
    """Generate initial demo sensor data"""
    print("📊 Generating initial demo data...")
    
    # Generate 20 demo data points
    base_time = datetime.now()
    for i in range(20):
        temp = random.uniform(22.0, 28.0)
        hum = random.uniform(45.0, 75.0)
        
        st.session_state.sensor_data.append({
            'timestamp': base_time - timedelta(minutes=(19-i) * 5),  # 5 minute intervals
            'temperature': round(temp, 1),
            'humidity': round(hum, 1),
            'source': 'demo',
            'device_id': 'demo_sensor_01',
            'topic': 'demo'
        })
    
    # Generate initial predictions
    if st.session_state.sensor_data:
        latest = st.session_state.sensor_data[-1]
        make_prediction_local(latest['temperature'], latest['humidity'])
    
    print(f"✅ Generated {len(st.session_state.sensor_data)} demo data points")

# ==================== SIMULATED FUNCTIONS ====================
def load_all_models():
    """Simulasi load models"""
    st.session_state.ml_models = {
        'Decision Tree': {'type': 'Classifier', 'accuracy': 0.85, 'color': '#3498DB'},
        'KNN': {'type': 'Classifier', 'accuracy': 0.82, 'color': '#2ECC71'},
        'Logistic Regression': {'type': 'Classifier', 'accuracy': 0.80, 'color': '#9B59B6'},
        'Random Forest': {'type': 'Classifier', 'accuracy': 0.88, 'color': '#E74C3C'},
        'SVM': {'type': 'Classifier', 'accuracy': 0.83, 'color': '#F39C12'}
    }
    return True

def make_prediction_local(temperature, humidity):
    """Simulasi prediksi"""
    if not st.session_state.ml_models:
        print("⚠️ No models loaded for prediction")
        return {}
    
    print(f"🧠 Making prediction for Temp={temperature}, Humid={humidity}")
    
    predictions = {}
    
    for model_name, model_info in st.session_state.ml_models.items():
        # Different prediction logic for each model
        if 'Decision' in model_name:
            if temperature < 22:
                label = 'COLD'
                confidence = 0.85 + random.uniform(-0.05, 0.05)
            elif temperature > 26:
                label = 'HOT'
                confidence = 0.90 + random.uniform(-0.05, 0.05)
            else:
                label = 'NORMAL'
                confidence = 0.95 + random.uniform(-0.05, 0.05)
                
        elif 'KNN' in model_name:
            if temperature < 21:
                label = 'VERY COLD'
                confidence = 0.80 + random.uniform(-0.05, 0.05)
            elif temperature > 27:
                label = 'VERY HOT'
                confidence = 0.85 + random.uniform(-0.05, 0.05)
            else:
                label = 'NORMAL'
                confidence = 0.88 + random.uniform(-0.05, 0.05)
                
        elif 'Logistic' in model_name:
            if temperature < 23:
                label = 'LOW'
                confidence = 0.75 + random.uniform(-0.05, 0.05)
            elif temperature > 25:
                label = 'HIGH'
                confidence = 0.78 + random.uniform(-0.05, 0.05)
            else:
                label = 'MEDIUM'
                confidence = 0.82 + random.uniform(-0.05, 0.05)
                
        elif 'Forest' in model_name:
            # Random Forest specific logic
            score = (temperature - 20) / 10  # Normalize
            if score < 0.3:
                label = 'COLD'
                confidence = 0.88 + random.uniform(-0.03, 0.03)
            elif score > 0.7:
                label = 'HOT'
                confidence = 0.92 + random.uniform(-0.03, 0.03)
            else:
                label = 'OPTIMAL'
                confidence = 0.96 + random.uniform(-0.03, 0.03)
                
        else:  # SVM
            if temperature < 24:
                label = 'COOL'
                confidence = 0.83 + random.uniform(-0.05, 0.05)
            elif temperature > 26:
                label = 'WARM'
                confidence = 0.86 + random.uniform(-0.05, 0.05)
            else:
                label = 'IDEAL'
                confidence = 0.89 + random.uniform(-0.05, 0.05)
        
        predictions[model_name] = {
            'label': label,
            'confidence': round(confidence, 3),
            'model_type': model_info['type'],
            'color': model_info.get('color', '#1E88E5')
        }
    
    history_entry = {
        'timestamp': datetime.now(),
        'temperature': temperature,
        'humidity': humidity,
        'predictions': predictions,
        'type': 'ml_prediction'
    }
    st.session_state.predictions.append(history_entry)
    st.session_state.last_prediction_time = datetime.now()
    
    if len(st.session_state.predictions) > 50:
        st.session_state.predictions = st.session_state.predictions[-50:]
    
    print(f"✅ Prediction made with {len(predictions)} models")
    return predictions

def generate_sample_sensor_data():
    """Generate sample sensor data untuk demo"""
    print("📊 Generating sample sensor data...")
    
    for i in range(5):
        temp = random.uniform(20, 30)
        hum = random.uniform(40, 80)
        
        st.session_state.sensor_data.append({
            'timestamp': datetime.now() - timedelta(minutes=4-i),
            'temperature': round(temp, 2),
            'humidity': round(hum, 2),
            'source': 'demo',
            'device_id': 'demo_sensor_01'
        })
    
    print(f"✅ Added {5} new demo data points")

# ==================== SIDEBAR ====================
def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
        
        # MQTT Connection Control
        st.markdown("---")
        st.subheader("📡 MQTT Connection")
        
        client = st.session_state.mqtt_client
        stats = client.get_connection_stats()
        
        # Connection Status with icon
        status_col1, status_col2 = st.columns([1, 3])
        with status_col1:
            if client.connected:
                st.markdown('<div style="text-align: center;">✅</div>', unsafe_allow_html=True)
            elif client.connecting:
                st.markdown('<div style="text-align: center;">🔄</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="text-align: center;">❌</div>', unsafe_allow_html=True)
        
        with status_col2:
            if client.connected:
                st.markdown('<p class="status-connected">Connected to HiveMQ</p>', unsafe_allow_html=True)
                if st.session_state.mqtt_connection_time:
                    connect_time = st.session_state.mqtt_connection_time.strftime('%H:%M:%S')
                    st.caption(f"Since: {connect_time}")
            elif client.connecting:
                st.markdown('<p class="status-connecting">Connecting...</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-disconnected">Disconnected</p>', unsafe_allow_html=True)
        
        # Connection Buttons
        col1, col2 = st.columns(2)
        with col1:
            if not client.connected and not client.connecting:
                if st.button("🔗 Connect", use_container_width=True, type="primary"):
                    with st.spinner("Connecting to HiveMQ Cloud..."):
                        client.connect_async()
                        time.sleep(1)  # Small delay for connection attempt
                        st.rerun()
            elif client.connecting:
                st.button("🔄 Connecting...", use_container_width=True, disabled=True)
            else:
                st.button("✅ Connected", use_container_width=True, disabled=True)
        
        with col2:
            if client.connected or client.connecting:
                if st.button("🔌 Disconnect", use_container_width=True):
                    client.disconnect()
                    time.sleep(0.5)
                    st.rerun()
            else:
                st.button("🔌 Disconnect", use_container_width=True, disabled=True)
        
        # Quick Stats
        with st.expander("📊 Quick Stats", expanded=True):
            st.metric("📨 Messages", stats['messages_received'])
            st.metric("📤 Published", stats['messages_published'])
            st.metric("📈 Sensor Data", len(st.session_state.sensor_data))
            st.metric("🤖 Predictions", len(st.session_state.predictions))
            
            if st.session_state.last_mqtt_msg:
                last_time = st.session_state.last_mqtt_msg['timestamp'].strftime('%H:%M:%S')
                st.caption(f"Last message: {last_time}")
        
        # Model Management
        st.markdown("---")
        st.subheader("🤖 Model Management")
        
        if st.button("🔄 Load All Models", use_container_width=True, type="secondary"):
            with st.spinner("Loading machine learning models..."):
                if load_all_models():
                    st.success(f"✅ Loaded {len(st.session_state.ml_models)} models")
                    time.sleep(1)
                    st.rerun()
        
        if st.session_state.ml_models:
            st.write("**Loaded Models:**")
            for name, info in st.session_state.ml_models.items():
                with st.container():
                    cols = st.columns([3, 1])
                    with cols[0]:
                        st.caption(f"• {name}")
                    with cols[1]:
                        st.caption(f"{info['accuracy']:.0%}")
        
        # Auto Features
        st.markdown("---")
        st.subheader("⚡ Auto Features")
        
        col_auto1, col_auto2 = st.columns(2)
        with col_auto1:
            auto_predict = st.toggle("Auto Predict", 
                                    value=st.session_state.auto_predict,
                                    help="Automatically make predictions when new data arrives")
            if auto_predict != st.session_state.auto_predict:
                st.session_state.auto_predict = auto_predict
                st.rerun()
        
        with col_auto2:
            auto_refresh = st.toggle("Auto Refresh", 
                                    value=True,
                                    help="Auto-refresh dashboard every 2 seconds")
        
        # Demo Controls
        st.markdown("---")
        st.subheader("🎯 Demo Controls")
        
        # Manual prediction controls
        st.write("**Manual Prediction:**")
        col_temp, col_hum = st.columns(2)
        with col_temp:
            temp_input = st.number_input("Temp (°C)", 
                                        min_value=15.0, 
                                        max_value=35.0, 
                                        value=25.0, 
                                        step=0.5,
                                        key="temp_input")
        with col_hum:
            hum_input = st.number_input("Humid (%)", 
                                       min_value=30.0, 
                                       max_value=90.0, 
                                       value=65.0, 
                                       step=1.0,
                                       key="hum_input")
        
        if st.button("🧠 Make Prediction", use_container_width=True):
            if st.session_state.ml_models:
                predictions = make_prediction_local(temp_input, hum_input)
                
                # Publish to MQTT if connected
                if client.connected:
                    prediction_msg = {
                        'timestamp': datetime.now().isoformat(),
                        'temperature': temp_input,
                        'humidity': hum_input,
                        'predictions': predictions,
                        'source': 'manual'
                    }
                    client.publish(MQTT_TOPIC_PUBLISH, json.dumps(prediction_msg))
                    st.toast("✅ Prediction published to MQTT!", icon="📤")
                
                st.success(f"Prediction made with {len(predictions)} models")
                st.rerun()
            else:
                st.warning("Please load models first")
        
        # Quick action buttons
        col_demo1, col_demo2 = st.columns(2)
        with col_demo1:
            if st.button("📊 Add Demo Data", use_container_width=True):
                generate_sample_sensor_data()
                st.rerun()
        
        with col_demo2:
            if st.button("📤 Test MQTT", use_container_width=True):
                if client.connected:
                    test_msg = {
                        'temperature': round(random.uniform(20, 30), 2),
                        'humidity': round(random.uniform(40, 80), 2),
                        'device_id': 'streamlit_test',
                        'timestamp': datetime.now().isoformat(),
                        'test': True,
                        'message': 'Test message from Streamlit Dashboard'
                    }
                    
                    if client.publish(MQTT_TOPIC_SUBSCRIBE, json.dumps(test_msg)):
                        st.toast("✅ Test message sent!", icon="✅")
                    else:
                        st.toast("❌ Failed to send test message", icon="❌")
                else:
                    st.warning("Not connected to MQTT")
        
        # Clear Data
        st.markdown("---")
        if st.button("🗑️ Clear All Data", use_container_width=True, type="secondary"):
            st.session_state.sensor_data = []
            st.session_state.predictions = []
            st.session_state.mqtt_messages = []
            generate_initial_demo_data()  # Regenerate demo data
            st.success("Data cleared and regenerated!")
            st.rerun()
        
        # Footer
        st.markdown("---")
        st.caption(f"Client ID: {MQTT_CLIENT_ID[:15]}...")
        st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")

# ==================== MAIN DASHBOARD ====================
def render_dashboard():
    # Header
    st.markdown("<h1 class='main-title'>🤖 IoT ML Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #666;'>Real-time MQTT + Machine Learning Integration</h4>", unsafe_allow_html=True)
    
    # Top Metrics Row
    st.markdown("---")
    
    # Get latest sensor data
    latest_temp = "N/A"
    latest_hum = "N/A"
    latest_source = "No data"
    
    if st.session_state.sensor_data:
        latest = st.session_state.sensor_data[-1]
        latest_temp = f"{latest['temperature']:.1f}°C"
        latest_hum = f"{latest['humidity']:.1f}%"
        latest_source = latest.get('source', 'unknown')
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        client = st.session_state.mqtt_client
        if client.connected:
            st.metric("📡 Status", "CONNECTED", delta="Live")
        elif client.connecting:
            st.metric("📡 Status", "CONNECTING", delta="...")
        else:
            st.metric("📡 Status", "DISCONNECTED", delta="Offline", delta_color="off")
    
    with col2:
        st.metric("🌡️ Temperature", latest_temp)
    
    with col3:
        st.metric("💧 Humidity", latest_hum)
    
    with col4:
        st.metric("🤖 Models", len(st.session_state.ml_models))
    
    with col5:
        messages_received = st.session_state.mqtt_client.stats['messages_received']
        st.metric("📨 Messages", messages_received)
    
    # Data Source Info
    if st.session_state.sensor_data:
        source_colors = {
            'mqtt': '#2ECC71',
            'demo': '#3498DB',
            'manual': '#9B59B6'
        }
        source_color = source_colors.get(latest_source, '#95A5A6')
        st.caption(f"Latest data source: <span style='color:{source_color}; font-weight:bold'>{latest_source.upper()}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["📡 Live Dashboard", "📈 Sensor Data", "🔮 Predictions"])
    
    with tab1:
        # Live Dashboard
        col_live1, col_live2 = st.columns([2, 1])
        
        with col_live1:
            st.subheader("📊 Real-time Charts")
            
            if st.session_state.sensor_data:
                # Convert to DataFrame for charts
                df = pd.DataFrame(st.session_state.sensor_data)
                
                # Temperature Chart
                st.write("**Temperature Trend**")
                if len(df) > 1:
                    chart_df = df.tail(20).copy()
                    chart_df['time'] = chart_df['timestamp'].dt.strftime('%H:%M')
                    
                    # Create two charts side by side
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        st.line_chart(chart_df.set_index('time')[['temperature']])
                    
                    with chart_col2:
                        st.line_chart(chart_df.set_index('time')[['humidity']])
                else:
                    st.info("Need more data points for charts")
            
            # Latest MQTT Messages
            st.subheader("📨 Latest MQTT Messages")
            
            if st.session_state.mqtt_messages:
                # Show last 5 messages
                for msg in reversed(st.session_state.mqtt_messages[-5:]):
                    with st.expander(f"📡 {msg['topic']} - {msg['timestamp'].strftime('%H:%M:%S')}"):
                        if isinstance(msg['data'], dict):
                            st.json(msg['data'], expanded=False)
                        else:
                            st.code(msg['data'])
                        
                        # Show extracted values
                        if 'temperature' in str(msg['data']) or 'humidity' in str(msg['data']):
                            st.caption("✅ Contains sensor data")
            else:
                st.info("No MQTT messages received yet")
        
        with col_live2:
            st.subheader("⚡ Quick Actions")
            
            # Connection Test
            if st.button("🔗 Test Connection", use_container_width=True):
                client = st.session_state.mqtt_client
                if client.connected:
                    st.success("✅ Connected to HiveMQ")
                    st.caption(f"Messages received: {client.stats['messages_received']}")
                else:
                    st.error("❌ Not connected")
            
            # Add Random Data
            if st.button("🎲 Add Random Data", use_container_width=True):
                temp = round(random.uniform(20, 30), 2)
                hum = round(random.uniform(40, 80), 2)
                
                st.session_state.sensor_data.append({
                    'timestamp': datetime.now(),
                    'temperature': temp,
                    'humidity': hum,
                    'source': 'manual',
                    'device_id': 'manual_input'
                })
                
                # Auto predict if enabled
                if st.session_state.auto_predict:
                    make_prediction_local(temp, hum)
                
                st.success(f"Added: {temp}°C, {hum}%")
                st.rerun()
            
            # Predict Latest
            if st.session_state.sensor_data:
                if st.button("🧠 Predict Latest", use_container_width=True):
                    latest = st.session_state.sensor_data[-1]
                    make_prediction_local(latest['temperature'], latest['humidity'])
                    st.success("Prediction made!")
                    st.rerun()
            
            st.markdown("---")
            st.subheader("📈 Current Stats")
            
            stats_df = pd.DataFrame({
                'Metric': ['Sensor Data', 'Predictions', 'MQTT Messages', 'Models Loaded'],
                'Value': [
                    len(st.session_state.sensor_data),
                    len(st.session_state.predictions),
                    st.session_state.mqtt_client.stats['messages_received'],
                    len(st.session_state.ml_models)
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with tab2:
        # Sensor Data Tab
        st.subheader("📈 Sensor Data History")
        
        if st.session_state.sensor_data:
            # Convert to DataFrame
            df = pd.DataFrame(st.session_state.sensor_data)
            
            # Sort by timestamp
            df = df.sort_values('timestamp', ascending=False)
            
            # Display in a nice table
            st.dataframe(
                df.head(20),
                use_container_width=True,
                column_config={
                    'timestamp': st.column_config.DatetimeColumn(
                        label="Time",
                        format="HH:mm:ss",
                        width="small"
                    ),
                    'temperature': st.column_config.NumberColumn(
                        label="Temperature (°C)",
                        format="%.1f",
                        width="small"
                    ),
                    'humidity': st.column_config.NumberColumn(
                        label="Humidity (%)",
                        format="%.1f",
                        width="small"
                    ),
                    'source': st.column_config.TextColumn(
                        label="Source",
                        width="small"
                    ),
                    'device_id': st.column_config.TextColumn(
                        label="Device ID",
                        width="medium"
                    )
                }
            )
            
            # Statistics
            st.subheader("📊 Statistics")
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                if len(df) > 0:
                    avg_temp = df['temperature'].mean()
                    st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
            
            with stat_col2:
                if len(df) > 0:
                    avg_hum = df['humidity'].mean()
                    st.metric("Avg Humidity", f"{avg_hum:.1f}%")
            
            with stat_col3:
                sources = df['source'].unique()
                st.metric("Data Sources", len(sources))
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        else:
            st.info("No sensor data available. Add some data using the demo controls.")
    
    with tab3:
        # Predictions Tab
        st.subheader("🔮 Prediction Results")
        
        if st.session_state.predictions:
            latest_pred = st.session_state.predictions[-1]
            
            # Display latest prediction prominently
            st.markdown(f"""
            <div class="sensor-data-card">
                <h3 style="color: white; text-align: center;">Latest Prediction</h3>
                <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                    <div style="text-align: center;">
                        <h4 style="margin: 0; color: #FFD700;">🌡️ Temperature</h4>
                        <h2 style="margin: 5px 0; color: white;">{latest_pred['temperature']:.1f}°C</h2>
                    </div>
                    <div style="text-align: center;">
                        <h4 style="margin: 0; color: #87CEEB;">💧 Humidity</h4>
                        <h2 style="margin: 5px 0; color: white;">{latest_pred['humidity']:.1f}%</h2>
                    </div>
                </div>
                <p style="text-align: center; color: rgba(255,255,255,0.9);">
                    {latest_pred['timestamp'].strftime('%H:%M:%S')} | {latest_pred['type']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Model Predictions
            st.subheader("🤖 Model Predictions")
            
            if latest_pred['predictions']:
                # Create columns for model predictions
                pred_items = list(latest_pred['predictions'].items())
                
                # Display in a grid
                cols = st.columns(min(3, len(pred_items)))
                
                for idx, (model_name, pred_info) in enumerate(pred_items):
                    col_idx = idx % 3
                    with cols[col_idx]:
                        color = pred_info.get('color', '#1E88E5')
                        confidence_color = '#2ECC71' if pred_info['confidence'] > 0.8 else '#F39C12' if pred_info['confidence'] > 0.6 else '#E74C3C'
                        
                        st.markdown(f"""
                        <div class="prediction-card" style="border-left-color: {color};">
                            <h4 style="color: {color}; margin-bottom: 10px;">{model_name}</h4>
                            <h2 style="color: {color}; text-align: center; margin: 10px 0;">{pred_info['label']}</h2>
                            <div style="text-align: center; margin: 15px 0;">
                                <span style="color: {confidence_color}; font-size: 1.2em; font-weight: bold;">
                                    {pred_info['confidence']:.1%}
                                </span>
                                <br>
                                <span style="color: #666; font-size: 0.9em;">confidence</span>
                            </div>
                            <p style="text-align: center; color: #666; font-size: 0.9em; margin-top: 10px;">
                                {pred_info['model_type']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Prediction History
            st.subheader("📜 Prediction History")
            
            if len(st.session_state.predictions) > 1:
                history_data = []
                for pred in st.session_state.predictions[-10:]:
                    row = {
                        'Time': pred['timestamp'].strftime('%H:%M:%S'),
                        'Temperature': f"{pred['temperature']:.1f}°C",
                        'Humidity': f"{pred['humidity']:.1f}%",
                        'Models': len(pred['predictions'])
                    }
                    
                    # Add main prediction label (from first model)
                    if pred['predictions']:
                        first_model = list(pred['predictions'].values())[0]
                        row['Prediction'] = first_model['label']
                    
                    history_data.append(row)
                
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            
        else:
            st.info("No predictions yet. Load models and make predictions using the sidebar controls.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 10px; margin-top: 20px;">
        <p style="margin: 5px 0;">
            <strong>🚀 IoT ML Dashboard</strong> | 
            Broker: <code>{MQTT_BROKER}</code> | 
            Client ID: <code>{MQTT_CLIENT_ID[:20]}...</code>
        </p>
        <p style="margin: 5px 0; color: #666;">
            📊 {len(st.session_state.sensor_data)} data points | 
            🤖 {len(st.session_state.predictions)} predictions | 
            📨 {st.session_state.mqtt_client.stats['messages_received']} MQTT messages
        </p>
        <p style="margin: 5px 0; color: #888; font-size: 0.9em;">
            Last update: {datetime.now().strftime('%H:%M:%S')} | Auto-refresh: Every 2 seconds
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN APP ====================
def main():
    # Initialize session state
    init_session_state()
    
    # Process MQTT messages if connected
    client = st.session_state.mqtt_client
    if client.connected:
        messages = client.get_messages()
        for msg in messages:
            # Messages are already processed in on_message callback
            # Just update the UI state
            st.session_state.mqtt_messages.append(msg)
            if len(st.session_state.mqtt_messages) > 50:
                st.session_state.mqtt_messages = st.session_state.mqtt_messages[-50:]
    
    # Render sidebar
    render_sidebar()
    
    # Render main dashboard
    render_dashboard()
    
    # Auto-refresh every 2 seconds
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
        
        Running in demo mode with simulated data...
        """)
        
        # Initialize session state for demo mode
        init_session_state()
        
        # Run in demo mode
        render_sidebar()
        render_dashboard()
        
        # Auto-refresh
        time.sleep(2)
        st.rerun()
