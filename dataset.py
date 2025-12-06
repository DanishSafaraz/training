import paho.mqtt.client as mqtt
import json
import pandas as pd
import time
from datetime import datetime
import os
import sys

# ==================== KONFIGURASI ====================
MQTT_BROKER = "48be83e63863499c87afce855025c93e.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764992629489"
MQTT_PASSWORD = ".09yUhd13*nZF?A#rjKT"
DHT_TOPIC = "iot/sensor/data"

# File Configuration
FOLDER_PATH = "Trainingdht"
CSV_FILENAME = "sensor_data.csv"
CSV_PATH = os.path.join(FOLDER_PATH, CSV_FILENAME)

# DataFrame untuk menyimpan data
sensor_data = pd.DataFrame(columns=['timestamp', 'temperature', 'humidity', 'label', 'label_encoded'])

# ==================== SETUP CSV DENGAN PANDAS ====================
def setup_csv():
    """Setup file CSV menggunakan Pandas"""
    global sensor_data
    
    if not os.path.exists(FOLDER_PATH):
        os.makedirs(FOLDER_PATH)
        print(f"📁 Created folder: {FOLDER_PATH}")
    
    # Cek apakah file sudah ada
    file_exists = os.path.exists(CSV_PATH)
    
    if file_exists:
        try:
            # Load data yang sudah ada
            sensor_data = pd.read_csv(
                CSV_PATH, 
                sep=';',
                encoding='utf-8'
            )
            print(f"✅ Loaded existing CSV file: {CSV_PATH}")
            print(f"📊 Existing records: {len(sensor_data)}")
            
            # Tampilkan 3 data terakhir
            if not sensor_data.empty:
                print("📄 Last 3 records:")
                last_3 = sensor_data.tail(3)
                for _, row in last_3.iterrows():
                    print(f"   {row['timestamp']} | {row['temperature']}°C | {row['humidity']}% | {row['label']}")
        
        except Exception as e:
            print(f"⚠️ Error reading existing file: {e}")
            print("🔄 Creating new file...")
            file_exists = False
    
    if not file_exists:
        # Buat DataFrame kosong dengan kolom yang benar
        sensor_data = pd.DataFrame(columns=['timestamp', 'temperature', 'humidity', 'label', 'label_encoded'])
        print(f"✅ Created new CSV structure")
    
    return sensor_data

def save_to_csv():
    """Simpan DataFrame ke CSV"""
    try:
        # Save ke CSV dengan separator ';'
        sensor_data.to_csv(
            CSV_PATH, 
            sep=';', 
            index=False, 
            encoding='utf-8'
        )
        print(f"💾 Data saved to CSV: {len(sensor_data)} records")
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")

# ==================== MQTT CALLBACKS ====================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to HiveMQ Cloud!")
        print(f"📡 Subscribing to topic: {DHT_TOPIC}")
        client.subscribe(DHT_TOPIC)
        print("⏳ Waiting for DHT data...")
        print("=" * 50)
    else:
        print(f"❌ Connection failed with code: {rc}")

def on_message(client, userdata, msg):
    global sensor_data
    
    try:
        # Parse JSON data
        data = json.loads(msg.payload.decode())
        temperature = data.get('temperature', 0)
        humidity = data.get('humidity', 0)
        
        # Get current time
        now = datetime.now()
        timestamp_str = now.strftime('%H;%M;%S')
        
        # Determine label
        if temperature < 22:
            label = "DINGIN"
            label_encoded = 0
        elif temperature > 25:
            label = "PANAS"
            label_encoded = 2
        else:
            label = "NORMAL"
            label_encoded = 1
        
        # Buat dictionary baru
        new_data = {
            'timestamp': timestamp_str,
            'temperature': round(float(temperature), 2),
            'humidity': round(float(humidity), 2),
            'label': label,
            'label_encoded': label_encoded
        }
        
        # Tambahkan ke DataFrame menggunakan concat (lebih efisien untuk batch kecil)
        new_df = pd.DataFrame([new_data])
        sensor_data = pd.concat([sensor_data, new_df], ignore_index=True)
        
        # Print to console
        print(f"📥 Data received:")
        print(f"   ⏰ Time: {timestamp_str}")
        print(f"   🌡️  Temp: {temperature}°C")
        print(f"   💧 Hum: {humidity}%")
        print(f"   🏷️  Label: {label}")
        print("-" * 40)
        
        # Tampilkan progress
        print(f"📊 Total records: {len(sensor_data)} / 15")
        
        # Simpan ke file setiap 5 data atau jika sudah mencapai target
        if len(sensor_data) % 5 == 0:
            save_to_csv()
        
        # Stop after 15 records
        if len(sensor_data) >= 15:
            print("\n🎯 Reached 15 records! Stopping...")
            # Simpan terakhir kali
            save_to_csv()
            client.disconnect()
            
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        import traceback
        traceback.print_exc()

# ==================== MAIN PROGRAM ====================
def main():
    global sensor_data
    
    print("\n" + "="*50)
    print("🌡️  DHT DATA COLLECTOR with PANDAS (15 Records)")
    print("="*50)
    
    # Setup CSV file
    sensor_data = setup_csv()
    
    # Create MQTT client
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set()
    
    # Set callbacks
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Connect to broker
    try:
        print("🔗 Connecting to HiveMQ...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    # Start collecting
    print("\n🎯 Target: 15 records")
    print("Press Ctrl+C to stop early")
    print("="*50)
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
        # Simpan data yang sudah terkumpul
        if not sensor_data.empty:
            save_to_csv()
    finally:
        client.disconnect()
        
        # Final summary dengan Pandas
        print("\n" + "="*50)
        print("📊 COLLECTION SUMMARY")
        print("="*50)
        
        if not sensor_data.empty and len(sensor_data) > 0:
            print(f"✅ Total records collected: {len(sensor_data)}")
            print(f"💾 File saved: {CSV_PATH}")
            print(f"📁 Full path: {os.path.abspath(CSV_PATH)}")
            
            # Show file size
            if os.path.exists(CSV_PATH):
                file_size = os.path.getsize(CSV_PATH)
                print(f"📦 File size: {file_size} bytes")
            
            # Show sample data menggunakan Pandas
            print(f"\n📄 Sample data (first 3):")
            print(sensor_data.head(3).to_string(index=False))
            
            # Statistical summary
            print(f"\n📈 Statistical Summary:")
            print(f"   Temperature - Min: {sensor_data['temperature'].min():.1f}°C, "
                  f"Max: {sensor_data['temperature'].max():.1f}°C, "
                  f"Avg: {sensor_data['temperature'].mean():.1f}°C")
            print(f"   Humidity - Min: {sensor_data['humidity'].min():.1f}%, "
                  f"Max: {sensor_data['humidity'].max():.1f}%, "
                  f"Avg: {sensor_data['humidity'].mean():.1f}%")
            
            # Count labels dengan Pandas
            label_counts = sensor_data['label'].value_counts()
            print(f"\n🏷️  Label distribution:")
            for label, count in label_counts.items():
                percentage = (count / len(sensor_data)) * 100
                print(f"   {label}: {count} ({percentage:.1f}%)")
            
            # Simpan juga versi JSON untuk analisis lebih lanjut
            json_path = CSV_PATH.replace('.csv', '_summary.json')
            summary = {
                'total_records': len(sensor_data),
                'temperature_stats': {
                    'min': float(sensor_data['temperature'].min()),
                    'max': float(sensor_data['temperature'].max()),
                    'mean': float(sensor_data['temperature'].mean()),
                    'std': float(sensor_data['temperature'].std())
                },
                'humidity_stats': {
                    'min': float(sensor_data['humidity'].min()),
                    'max': float(sensor_data['humidity'].max()),
                    'mean': float(sensor_data['humidity'].mean()),
                    'std': float(sensor_data['humidity'].std())
                },
                'label_distribution': label_counts.to_dict()
            }
            
            import json as json_module
            with open(json_path, 'w', encoding='utf-8') as f:
                json_module.dump(summary, f, indent=2)
            print(f"\n📊 Summary saved to: {json_path}")
            
        else:
            print("❌ No data collected!")
        
        print("\n💡 Next: Run 'python trainingmodel.py' to train ML models")
        print("="*50)

if __name__ == "__main__":
    main()