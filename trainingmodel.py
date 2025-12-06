import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           ConfusionMatrixDisplay)
import pickle
import os
import json
from datetime import datetime
import warnings
import paho.mqtt.client as mqtt
import time
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI ====================
BASE_DIR = r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar"
CSV_FILE = os.path.join(BASE_DIR, "Trainingdht", "sensor_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "Trainingdht", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "Trainingdht", "reports")

# MQTT Configuration
MQTT_BROKER = "48be83e63863499c87afce855025c93e.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764992629489"
MQTT_PASSWORD = ".09yUhd13*nZF?A#rjKT"
MQTT_TOPIC = "iot/model/ml"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ==================== MQTT MANAGER ====================
class MQTTManager:
    """Manage MQTT connection and publishing"""
    
    def __init__(self):
        self.client = mqtt.Client()
        self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.client.tls_set()
        self.connected = False
        
    def connect(self):
        """Connect to MQTT broker"""
        try:
            print(f"🔗 Connecting to MQTT: {MQTT_BROKER}:{MQTT_PORT}")
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            time.sleep(2)
            self.connected = True
            print("✅ MQTT Connected!")
            return True
        except Exception as e:
            print(f"❌ MQTT Connection failed: {e}")
            return False
    
    def publish_prediction(self, model_name, prediction_data):
        """Publish prediction from specific model"""
        if not self.connected:
            print(f"⚠️  MQTT not connected, skipping publish")
            return False
        
        try:
            # Add model name to data
            prediction_data['model'] = model_name
            prediction_data['publish_time'] = datetime.now().strftime('%H:%M:%S')
            
            payload = json.dumps(prediction_data)
            result = self.client.publish(MQTT_TOPIC, payload, qos=1)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"📤 [{model_name}] Published: {prediction_data['label']}")
                return True
            else:
                print(f"❌ [{model_name}] Publish failed")
                return False
                
        except Exception as e:
            print(f"❌ [{model_name}] Publish error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT"""
        if self.connected:
            self.client.disconnect()
            self.connected = False
            print("📡 MQTT Disconnected")

# ==================== LOAD & PREPARE DATA ====================
def load_and_prepare_data():
    print("📂 Loading DHT dataset...")
    print(f"📁 File: {CSV_FILE}")
    
    try:
        if not os.path.exists(CSV_FILE):
            print(f"❌ File not found!")
            return None
        
        df = pd.read_csv(CSV_FILE, delimiter=';')
        print(f"✅ Dataset loaded: {df.shape[0]} records")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    # Display info
    print("\n📊 Dataset Info:")
    print(df.head())
    
    # Check labels
    unique_labels = df['label'].unique()
    print(f"\n🏷️  Unique labels: {list(unique_labels)}")
    
    # Process timestamp
    df[['hour', 'minute', 'second']] = df['timestamp'].str.split(';', expand=True).astype(int)
    
    # Create synthetic data if needed
    if len(unique_labels) < 3:
        print("\n⚠️  Creating synthetic data for all 3 classes...")
        from sklearn.datasets import make_classification
        
        # Get real data stats
        real_temp_mean = df['temperature'].mean()
        real_hum_mean = df['humidity'].mean()
        
        # Generate synthetic data
        X_synth, y_synth = make_classification(
            n_samples=100,
            n_features=4,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Scale to realistic ranges
        X_synth[:, 0] = X_synth[:, 0] * 5 + real_temp_mean  # Temperature
        X_synth[:, 1] = X_synth[:, 1] * 10 + real_hum_mean  # Humidity
        X_synth[:, 2] = np.random.randint(0, 24, 100)      # Hour
        X_synth[:, 3] = np.random.randint(0, 60, 100)      # Minute
        
        # Create synthetic DataFrame
        synth_df = pd.DataFrame(X_synth, columns=['temperature', 'humidity', 'hour', 'minute'])
        synth_df['label_encoded'] = y_synth
        synth_df['label'] = synth_df['label_encoded'].map({0: 'DINGIN', 1: 'NORMAL', 2: 'PANAS'})
        synth_df['timestamp'] = '00;00;00'
        
        # Combine with real data
        df = pd.concat([df, synth_df], ignore_index=True)
        print(f"📈 Combined dataset: {len(df)} records")
    
    # Features and target
    X = df[['temperature', 'humidity', 'hour', 'minute']]
    y = df['label_encoded']
    
    print(f"\n🔧 Features shape: {X.shape}")
    print(f"📊 Class distribution:")
    print(df['label'].value_counts())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📈 Data split:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df, X_test

# ==================== TRAIN ALL MODELS ====================
def train_all_models(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("🤖 TRAINING 3 ML MODELS")
    print("="*60)
    
    # Define all models
    models = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            criterion='gini'
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean'
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='ovr',
            solver='liblinear'
        )
    }
    
    results = {}
    label_names = ['DINGIN', 'NORMAL', 'PANAS']
    
    for name, model in models.items():
        print(f"\n🏃 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Per-class metrics
        try:
            precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        except:
            precision_per_class = [0, 0, 0]
            recall_per_class = [0, 0, 0]
            f1_per_class = [0, 0, 0]
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   📊 F1-Score: {f1:.4f}")
        print(f"   🔍 Precision: {precision:.4f}")
        print(f"   📈 Recall: {recall:.4f}")
        
        # Show predictions count
        unique, counts = np.unique(y_pred, return_counts=True)
        print(f"   🎯 Predictions distribution:")
        for label_code, count in zip(unique, counts):
            label_name = label_names[label_code] if label_code < 3 else f'Class_{label_code}'
            print(f"      {label_name}: {count}")
    
    return results, label_names

# ==================== CREATE ALL VISUALIZATIONS ====================
def create_all_visualizations(results, X_test_df, y_test, label_names):
    print("\n📊 CREATING VISUALIZATIONS FOR ALL MODELS...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. CONFUSION MATRICES - 3 MODELS
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('CONFUSION MATRICES - ALL MODELS', fontsize=16, fontweight='bold')
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        cm = confusion_matrix(y_test, result['y_pred'])
        
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=label_names
        )
        
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        ax.set_title(f'{name}', fontweight='bold', fontsize=12)
        
        # Add accuracy text
        accuracy = result['accuracy']
        ax.text(0.95, -0.15, f'Accuracy: {accuracy:.3f}', 
                transform=ax.transAxes, ha='right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'all_confusion_matrices.png'), 
                dpi=300, bbox_inches='tight')
    print("✅ Saved: all_confusion_matrices.png")
    
    # 2. MODEL COMPARISON BAR CHART
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MODEL PERFORMANCE COMPARISON', fontsize=16, fontweight='bold')
    
    metrics_list = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
    
    for idx, (metric, title) in enumerate(zip(metrics_list, metric_names)):
        ax = axes[idx//2, idx%2]
        model_names = list(results.keys())
        scores = [results[model][metric] for model in model_names]
        
        bars = ax.bar(model_names, scores, color=colors, alpha=0.8)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'model_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print("✅ Saved: model_comparison.png")
    
    # 3. FEATURE IMPORTANCE (Decision Tree)
    if 'Decision Tree' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        dt_model = results['Decision Tree']['model']
        feature_names = ['Temperature', 'Humidity', 'Hour', 'Minute']
        importances = dt_model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        bars = ax.barh(np.array(feature_names)[indices], 
                      importances[indices], 
                      color='#FF6B6B', alpha=0.8)
        
        ax.set_title('DECISION TREE - FEATURE IMPORTANCE', fontweight='bold', fontsize=14)
        ax.set_xlabel('Importance Score', fontsize=12)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', va='center', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'feature_importance.png'), 
                    dpi=300, bbox_inches='tight')
        print("✅ Saved: feature_importance.png")
    
    # 4. RECALL PER CLASS COMPARISON
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(label_names))
    width = 0.25
    
    for i, (model_name, result) in enumerate(results.items()):
        recalls = result['recall_per_class']
        offset = width * i
        ax.bar(x + offset, recalls, width, label=model_name, alpha=0.8)
        
        # Add value labels
        for j, recall in enumerate(recalls):
            ax.text(j + offset, recall + 0.02, f'{recall:.3f}', 
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_title('RECALL PER CLASS - ALL MODELS', fontweight='bold', fontsize=14)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Recall Score', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(label_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'recall_per_class.png'), 
                dpi=300, bbox_inches='tight')
    print("✅ Saved: recall_per_class.png")
    
    # 5. SUMMARY TABLE
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Create summary data
    summary_data = []
    for name, result in results.items():
        summary_data.append([
            name,
            f"{result['accuracy']:.4f}",
            f"{result['precision']:.4f}",
            f"{result['recall']:.4f}",
            f"{result['f1_score']:.4f}",
            f"{result['cv_mean']:.4f} ±{result['cv_std']:.4f}"
        ])
    
    # Create table
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Accuracy']
    table = ax.table(cellText=summary_data,
                     colLabels=columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['#FF6B6B'] * len(columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    plt.title('MODEL PERFORMANCE SUMMARY', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(REPORTS_DIR, 'performance_summary.png'), 
                dpi=300, bbox_inches='tight')
    print("✅ Saved: performance_summary.png")
    
    plt.show()

# ==================== SAVE ALL MODELS ====================
def save_all_models(results, scaler):
    print("\n💾 SAVING ALL MODELS...")
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Saved: scaler.pkl")
    
    # Save all models individually
    for name, result in results.items():
        # Create filename
        filename = name.lower().replace(' ', '_') + '.pkl'
        model_path = os.path.join(MODELS_DIR, filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f)
        
        print(f"✅ Saved: {filename}")
    
    # Save ensemble model (all models together)
    ensemble_path = os.path.join(MODELS_DIR, 'all_models.pkl')
    with open(ensemble_path, 'wb') as f:
        pickle.dump(results, f)
    print("✅ Saved: all_models.pkl (ensemble)")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models_trained': list(results.keys()),
        'performance': {},
        'mqtt_topic': MQTT_TOPIC,
        'note': 'All 3 models trained: DT, KNN, LR'
    }
    
    for name, result in results.items():
        metadata['performance'][name] = {
            'accuracy': float(result['accuracy']),
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1_score': float(result['f1_score']),
            'cv_mean': float(result['cv_mean']),
            'cv_std': float(result['cv_std'])
        }
    
    metadata_path = os.path.join(MODELS_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("✅ Saved: metadata.json")
    
    return metadata

# ==================== TEST & PUBLISH TO MQTT ====================
def test_and_publish_all_models(results, scaler, mqtt_manager):
    print("\n" + "="*60)
    print("📤 TESTING & PUBLISHING ALL MODELS TO MQTT")
    print("="*60)
    
    # Test cases
    test_cases = [
        # (temp, hum, hour, minute, expected)
        (18.0, 75.0, 14, 30, "DINGIN"),
        (20.5, 70.0, 10, 15, "DINGIN"),
        (23.0, 65.0, 12, 45, "NORMAL"),
        (24.5, 60.0, 15, 20, "NORMAL"),
        (26.0, 55.0, 18, 10, "PANAS"),
        (28.0, 50.0, 20, 0, "PANAS"),
        (32.0, 45.0, 22, 30, "PANAS"),
    ]
    
    all_predictions = []
    
    for temp, hum, hour, minute, expected in test_cases:
        print(f"\n🌡️  Test: {temp}°C, {hum}%, {hour:02d}:{minute:02d} (Expected: {expected})")
        print("-" * 50)
        
        # Prepare features
        features = np.array([[temp, hum, hour, minute]])
        features_scaled = scaler.transform(features)
        
        test_predictions = []
        
        for model_name, result in results.items():
            model = result['model']
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = probabilities[prediction]
            else:
                confidence = 1.0
                probabilities = [0, 0, 0]
            
            # Map to label
            label_map = {0: 'DINGIN', 1: 'NORMAL', 2: 'PANAS'}
            label = label_map.get(prediction, 'UNKNOWN')
            
            # Prepare data
            prediction_data = {
                'label': label,
                'label_encoded': int(prediction),
                'temperature': float(temp),
                'humidity': float(hum),
                'hour': hour,
                'minute': minute,
                'confidence': float(confidence),
                'expected': expected,
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'is_correct': label == expected
            }
            
            # Publish to MQTT
            if mqtt_manager.connected:
                mqtt_manager.publish_prediction(model_name, prediction_data)
            
            # Store for display
            test_predictions.append({
                'model': model_name,
                'prediction': label,
                'confidence': confidence,
                'correct': label == expected
            })
            
            print(f"   [{model_name:20}] → {label:10} ({confidence:.1%}) {'✅' if label == expected else '❌'}")
        
        all_predictions.append(test_predictions)
    
    # Summary
    print("\n" + "="*60)
    print("📊 PUBLISHING SUMMARY")
    print("="*60)
    
    for model_name in results.keys():
        correct_count = 0
        total_count = 0
        
        for test_set in all_predictions:
            for pred in test_set:
                if pred['model'] == model_name:
                    total_count += 1
                    if pred['correct']:
                        correct_count += 1
        
        if total_count > 0:
            accuracy = correct_count / total_count
            print(f"{model_name:20}: {correct_count}/{total_count} correct ({accuracy:.1%})")
    
    print(f"\n📡 All predictions published to: {MQTT_TOPIC}")
    print("💡 Arduino can subscribe to see all 3 model predictions")

# ==================== MAIN PROGRAM ====================
def main():
    print("\n" + "="*60)
    print("🚀 ADVANCED ML TRAINING - 3 MODELS + MQTT")
    print("="*60)
    print(f"Models: Decision Tree, KNN, Logistic Regression")
    print(f"MQTT Topic: {MQTT_TOPIC}")
    print("="*60)
    
    # Initialize MQTT
    mqtt_manager = MQTTManager()
    mqtt_manager.connect()
    
    try:
        # 1. Load and prepare data
        data_result = load_and_prepare_data()
        if data_result is None:
            print("❌ Failed to load data")
            return
        
        X_train, X_test, y_train, y_test, scaler, df, X_test_df = data_result
        
        # 2. Train all 3 models
        results, label_names = train_all_models(X_train, X_test, y_train, y_test)
        
        # 3. Create visualizations
        create_all_visualizations(results, X_test_df, y_test, label_names)
        
        # 4. Save all models
        metadata = save_all_models(results, scaler)
        
        # 5. Test and publish to MQTT
        test_and_publish_all_models(results, scaler, mqtt_manager)
        
        # 6. Show final summary
        print("\n" + "="*60)
        print("🏆 TRAINING COMPLETE - ALL 3 MODELS")
        print("="*60)
        
        print(f"\n📊 Models trained and saved:")
        for model_name in results.keys():
            acc = results[model_name]['accuracy']
            f1 = results[model_name]['f1_score']
            print(f"   • {model_name:25} → Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        # Find best model by F1-score
        best_model = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
        best_f1 = results[best_model]['f1_score']
        
        print(f"\n🏆 Best Model: {best_model} (F1-Score: {best_f1:.4f})")
        
        print(f"\n📁 Files saved:")
        print(f"   • Models: {MODELS_DIR}/")
        print(f"   • Reports: {REPORTS_DIR}/")
        
        print(f"\n📡 MQTT Publishing:")
        print(f"   • Topic: {MQTT_TOPIC}")
        print(f"   • All 3 models publish predictions")
        print(f"   • Arduino can subscribe for real-time control")
        
        print(f"\n🔧 How to use models:")
        print(f"   1. Load with: pickle.load('model.pkl')")
        print(f"   2. Predict with: model.predict(features)")
        print(f"   3. Check metadata.json for performance details")
        
        print("\n✅ All 3 models ready for deployment!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Disconnect MQTT
        mqtt_manager.disconnect()

if __name__ == "__main__":
    main()