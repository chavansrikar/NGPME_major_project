import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import os
import ffmpeg
import bcrypt
import plotly.express as px
import plotly.graph_objects as go
import math
from scipy.signal import find_peaks, savgol_filter
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# --- DEEP LEARNING IMPORTS ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# --- PASSWORD AND DB FUNCTIONS ---
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def add_user(username, password_hash, db_name="gait_analysis.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username, db_name="gait_analysis.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    user_record = cursor.fetchone()
    conn.close()
    return user_record

def setup_database(db_name="gait_analysis.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY, 
                        username TEXT UNIQUE NOT NULL, 
                        password_hash TEXT NOT NULL
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY, 
                        username TEXT, 
                        timestamp TEXT, 
                        video_name TEXT, 
                        total_steps INTEGER,
                        cadence REAL,
                        avg_left_knee_angle REAL, 
                        avg_right_knee_angle REAL, 
                        asymmetry REAL, 
                        avg_pelvic_tilt REAL, 
                        pelvic_tilt_range REAL, 
                        avg_stride_length_px REAL,
                        stride_variability REAL,
                        avg_step_width REAL,
                        toe_clearance_min REAL,
                        gait_smoothness REAL,
                        balance_score REAL,
                        fall_risk TEXT,
                        final_risk_score REAL,
                        anomaly_detected INTEGER,
                        detected_issues TEXT,
                        FOREIGN KEY (username) REFERENCES users (username)
                    )''')
    conn.commit()
    conn.close()

def save_session_to_db(username, summary_data, db_name="gait_analysis.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO sessions 
                      (username, timestamp, video_name, total_steps, cadence, avg_left_knee_angle, 
                       avg_right_knee_angle, asymmetry, avg_pelvic_tilt, pelvic_tilt_range, 
                       avg_stride_length_px, stride_variability, avg_step_width, toe_clearance_min,
                       gait_smoothness, balance_score, fall_risk, final_risk_score, 
                       anomaly_detected, detected_issues) 
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                   (username, summary_data['timestamp'], summary_data['video_name'], 
                    summary_data['total_steps'], summary_data['cadence'],
                    summary_data['avg_left_knee_angle'], 
                    summary_data['avg_right_knee_angle'], summary_data['asymmetry'], 
                    summary_data['avg_pelvic_tilt'], summary_data['pelvic_tilt_range'], 
                    summary_data['avg_stride_length_px'], summary_data['stride_variability'],
                    summary_data['avg_step_width'], summary_data['toe_clearance_min'],
                    summary_data['gait_smoothness'], summary_data['balance_score'],
                    summary_data['fall_risk'], summary_data['final_risk_score'],
                    summary_data['anomaly_detected'], summary_data['detected_issues']))
    conn.commit()
    conn.close()

def load_user_history(username, db_name="gait_analysis.db"):
    conn = sqlite3.connect(db_name)
    history_df = pd.read_sql_query("SELECT * FROM sessions WHERE username = ? ORDER BY timestamp DESC", conn, params=(username,))
    conn.close()
    return history_df

# --- UTILITY FUNCTIONS ---
def calculate_angle(a, b, c):
    """Calculate angle at point b formed by points a-b-c"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_velocity(positions, fps):
    """Calculate velocity from position history"""
    if len(positions) < 2:
        return 0
    distances = [calculate_distance(positions[i], positions[i+1]) for i in range(len(positions)-1)]
    return np.mean(distances) * fps if distances else 0

# --- DEEP LEARNING MODEL ---
DL_MODEL = None
DL_SCALER = None
ANOMALY_DETECTOR = None
MODEL_PATH = "dl_gait_model.h5"
SCALER_PATH = "dl_scaler.joblib"
ANOMALY_PATH = "anomaly_detector.joblib"

def create_lstm_model(input_shape, output_shape):
    """Create advanced LSTM model for temporal gait analysis"""
    model = keras.Sequential([
        # LSTM layers for temporal pattern recognition
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.2),
        
        # Dense layers for classification
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_shape, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_and_train_dl_model():
    """Create and train deep learning model with synthetic data"""
    print("🧠 Creating Deep Learning model...")
    
    # Generate synthetic temporal gait sequences
    # Shape: (samples, timesteps, features)
    # Features: [left_knee, right_knee, pelvic_tilt, left_ankle_vel, right_ankle_vel, COM_x, COM_y]
    
    def generate_gait_sequence(pattern_type, length=50):
        """Generate realistic gait sequence"""
        time = np.linspace(0, 4*np.pi, length)
        
        if pattern_type == 0:  # Normal gait - MOST COMMON
            left_knee = 165 + 12 * np.sin(time)
            right_knee = 165 + 12 * np.sin(time + np.pi)
            pelvic_tilt = 90 + 2 * np.sin(time * 0.5)  # Near level
            noise = 0.8
        elif pattern_type == 1:  # Asymmetric gait - MODERATE
            left_knee = 165 + 12 * np.sin(time)
            right_knee = 160 + 10 * np.sin(time + np.pi + 0.2)  # Slightly off
            pelvic_tilt = 90 + 6 * np.sin(time * 0.5)  # More tilt
            noise = 1.2
        elif pattern_type == 2:  # Shuffling gait - CONCERNING
            left_knee = 170 + 7 * np.sin(time)  # Reduced ROM
            right_knee = 170 + 7 * np.sin(time + np.pi)
            pelvic_tilt = 90 + 4 * np.sin(time * 0.5)
            noise = 1.0
        else:  # High risk gait - RARE
            left_knee = 165 + 15 * np.sin(time) + 3 * np.random.randn(length)
            right_knee = 155 + 8 * np.sin(time + np.pi + 0.4)
            pelvic_tilt = 90 + 10 * np.sin(time * 0.5)
            noise = 2.0
        
        # Add realistic noise
        left_knee += np.random.randn(length) * noise
        right_knee += np.random.randn(length) * noise
        pelvic_tilt += np.random.randn(length) * noise * 0.5
        
        # Clip to realistic ranges
        left_knee = np.clip(left_knee, 60, 180)
        right_knee = np.clip(right_knee, 60, 180)
        pelvic_tilt = np.clip(pelvic_tilt, 70, 110)
        
        # Velocities (derivatives)
        left_vel = np.gradient(left_knee)
        right_vel = np.gradient(right_knee)
        
        # Center of mass oscillation (smaller = better balance)
        com_multiplier = 2 if pattern_type == 0 else 5 if pattern_type < 3 else 8
        com_x = np.sin(time * 0.5) * com_multiplier
        com_y = np.abs(np.sin(time)) * (com_multiplier * 0.5)
        
        return np.column_stack([left_knee, right_knee, pelvic_tilt, left_vel, right_vel, com_x, com_y])
    
    # Generate training data with realistic distribution
    X_train = []
    y_train = []
    
    # More normal samples (real-world distribution)
    samples_per_class = [150, 80, 50, 30]  # Normal:Asymmetric:Shuffling:HighRisk
    
    for class_label in range(4):
        for _ in range(samples_per_class[class_label]):
            sequence = generate_gait_sequence(class_label, length=50)
            X_train.append(sequence)
            y_train.append(class_label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Create scaler for features
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    
    # Create and train model
    model = create_lstm_model(input_shape=(50, 7), output_shape=4)
    
    # Train with early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss', patience=10, restore_best_weights=True
    )
    
    model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=16,
        verbose=0,
        callbacks=[early_stopping]
    )
    
    # Save model and scaler
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print("✅ Deep Learning model trained and saved!")
    return model, scaler

def create_anomaly_detector():
    """Create Isolation Forest for anomaly detection"""
    print("🔍 Creating Anomaly Detector...")
    
    # Generate REALISTIC normal gait patterns (more lenient)
    normal_features = []
    
    # Good gait (most common)
    for _ in range(150):
        asymmetry = np.random.uniform(0, 10)  # Up to 10° is normal
        stride_var = np.random.uniform(0, 15)  # Some variation is normal
        pelvic_range = np.random.uniform(0, 10)  # Up to 10° is normal
        cadence = np.random.uniform(90, 125)  # Normal range
        step_width = np.random.uniform(6, 16)  # Normal range
        toe_clear = np.random.uniform(2, 10)  # Safe clearance
        normal_features.append([asymmetry, stride_var, pelvic_range, cadence, step_width, toe_clear])
    
    # Moderate concerns (still fairly common)
    for _ in range(80):
        asymmetry = np.random.uniform(8, 15)
        stride_var = np.random.uniform(12, 20)
        pelvic_range = np.random.uniform(8, 15)
        cadence = np.random.uniform(85, 100)
        step_width = np.random.uniform(4, 8)
        toe_clear = np.random.uniform(1.5, 4)
        normal_features.append([asymmetry, stride_var, pelvic_range, cadence, step_width, toe_clear])
    
    # Add true anomalies (rare)
    for _ in range(20):
        asymmetry = np.random.uniform(18, 30)
        stride_var = np.random.uniform(25, 40)
        pelvic_range = np.random.uniform(18, 30)
        cadence = np.random.uniform(60, 80)
        step_width = np.random.uniform(1, 5)
        toe_clear = np.random.uniform(0, 1.5)
        normal_features.append([asymmetry, stride_var, pelvic_range, cadence, step_width, toe_clear])
    
    X = np.array(normal_features)
    
    # Train Isolation Forest with lower contamination (fewer false positives)
    detector = IsolationForest(contamination=0.08, random_state=42)
    detector.fit(X)
    
    joblib.dump(detector, ANOMALY_PATH)
    print("✅ Anomaly detector trained and saved!")
    return detector

def initialize_models():
    """Initialize all ML/DL models"""
    global DL_MODEL, DL_SCALER, ANOMALY_DETECTOR
    
    if DL_MODEL is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            print("📥 Loading existing Deep Learning model...")
            DL_MODEL = keras.models.load_model(MODEL_PATH)
            DL_SCALER = joblib.load(SCALER_PATH)
        else:
            DL_MODEL, DL_SCALER = create_and_train_dl_model()
    
    if ANOMALY_DETECTOR is None:
        if os.path.exists(ANOMALY_PATH):
            print("📥 Loading existing Anomaly Detector...")
            ANOMALY_DETECTOR = joblib.load(ANOMALY_PATH)
        else:
            ANOMALY_DETECTOR = create_anomaly_detector()

def prepare_sequence_for_dl(gait_data_df, sequence_length=50):
    """Prepare gait sequence for DL model"""
    if len(gait_data_df) < sequence_length:
        # Pad if too short
        padding = sequence_length - len(gait_data_df)
        gait_data_df = pd.concat([gait_data_df] * (padding // len(gait_data_df) + 1))
        gait_data_df = gait_data_df.iloc[:sequence_length]
    else:
        # Take last sequence_length frames
        gait_data_df = gait_data_df.iloc[-sequence_length:]
    
    # Extract features
    left_knee = gait_data_df['left_knee_smoothed'].values
    right_knee = gait_data_df['right_knee_smoothed'].values
    pelvic_tilt = gait_data_df['pelvic_tilt_smoothed'].values
    
    # Calculate velocities
    left_vel = np.gradient(left_knee)
    right_vel = np.gradient(right_knee)
    
    # COM movement (simplified)
    com_x = gait_data_df['com_x'].values if 'com_x' in gait_data_df else np.zeros(sequence_length)
    com_y = gait_data_df['com_y'].values if 'com_y' in gait_data_df else np.zeros(sequence_length)
    
    # Combine features
    sequence = np.column_stack([left_knee, right_knee, pelvic_tilt, left_vel, right_vel, com_x, com_y])
    
    return sequence

def assess_fall_risk_dl(gait_data_df, asymmetry, stride_variability, pelvic_tilt_range, 
                        cadence, step_width, toe_clearance):
    """Advanced fall risk assessment using Deep Learning + Anomaly Detection"""
    initialize_models()
    
    # Prepare sequence for LSTM
    sequence = prepare_sequence_for_dl(gait_data_df)
    sequence_scaled = DL_SCALER.transform(sequence.reshape(-1, 7)).reshape(1, 50, 7)
    
    # Get DL prediction
    dl_predictions = DL_MODEL.predict(sequence_scaled, verbose=0)[0]
    dl_class = np.argmax(dl_predictions)
    dl_confidence = dl_predictions[dl_class]
    
    # Map DL classes
    class_names = ["Normal", "Asymmetric", "Shuffling", "High Risk"]
    pattern_type = class_names[dl_class]
    
    # Anomaly detection
    feature_vector = np.array([[asymmetry, stride_variability, pelvic_tilt_range, 
                                cadence, step_width, toe_clearance]])
    anomaly_score = ANOMALY_DETECTOR.decision_function(feature_vector)[0]
    is_anomaly = ANOMALY_DETECTOR.predict(feature_vector)[0] == -1
    
    # Combine predictions for final risk score
    # Base score from DL class
    base_scores = {0: 15, 1: 45, 2: 60, 3: 85}
    risk_score = base_scores[dl_class]
    
    # Adjust based on confidence
    risk_score = risk_score * (0.7 + 0.3 * dl_confidence)
    
    # Adjust based on anomaly detection
    if is_anomaly:
        risk_score = min(100, risk_score + 20)
        anomaly_detected = 1
    else:
        anomaly_detected = 0
    
    # Additional adjustments based on specific metrics
    if asymmetry > 15:
        risk_score += 10
    if stride_variability > 18:
        risk_score += 8
    if cadence < 85:
        risk_score += 7
    if toe_clearance < 2:
        risk_score += 12
    
    risk_score = np.clip(risk_score, 0, 100)
    
    # Categorize risk
    if risk_score >= 75:
        category = "High"
    elif risk_score >= 50:
        category = "Moderate"
    elif risk_score >= 25:
        category = "Low"
    else:
        category = "Very Low"
    
    return category, round(risk_score, 2), pattern_type, anomaly_detected, dl_confidence

def detect_gait_issues(gait_data_df, asymmetry, stride_variability, cadence, 
                       toe_clearance, pelvic_range, step_width):
    """Detect specific gait problems with detailed feedback"""
    issues = []
    
    # Knee asymmetry analysis
    if asymmetry > 15:
        issues.append(f"⚠️ SEVERE ASYMMETRY: {asymmetry:.1f}° difference between knees")
    elif asymmetry > 10:
        issues.append(f"⚠️ MODERATE ASYMMETRY: {asymmetry:.1f}° knee angle difference")
    elif asymmetry > 7:
        issues.append(f"⚡ MILD ASYMMETRY: {asymmetry:.1f}° - Monitor progress")
    
    # Stride consistency
    if stride_variability > 20:
        issues.append(f"⚠️ HIGHLY VARIABLE STRIDE: SD={stride_variability:.1f}px - Poor consistency")
    elif stride_variability > 15:
        issues.append(f"⚡ INCONSISTENT STRIDE: SD={stride_variability:.1f}px")
    
    # Cadence analysis
    if cadence < 80:
        issues.append(f"⚠️ VERY SLOW CADENCE: {cadence:.0f} steps/min - Fall risk increased")
    elif cadence < 90:
        issues.append(f"⚡ SLOW CADENCE: {cadence:.0f} steps/min - Below normal range")
    elif cadence > 130:
        issues.append(f"⚡ FAST CADENCE: {cadence:.0f} steps/min - Possible rushing")
    
    # Toe clearance (trip risk)
    if toe_clearance < 2:
        issues.append(f"🚨 CRITICAL: Low toe clearance ({toe_clearance:.1f}px) - HIGH TRIP RISK!")
    elif toe_clearance < 4:
        issues.append(f"⚠️ LOW TOE CLEARANCE: {toe_clearance:.1f}px - Moderate trip risk")
    
    # Pelvic stability
    if pelvic_range > 15:
        issues.append(f"⚠️ EXCESSIVE PELVIC DROP: {pelvic_range:.1f}° - Weak hip abductors")
    elif pelvic_range > 10:
        issues.append(f"⚡ MODERATE PELVIC TILT: {pelvic_range:.1f}°")
    
    # Step width (balance)
    if step_width < 6:
        issues.append(f"⚠️ NARROW STEP WIDTH: {step_width:.1f}px - Balance concern")
    elif step_width > 18:
        issues.append(f"⚡ WIDE STEP WIDTH: {step_width:.1f}px - Possible compensation")
    
    # Gait smoothness analysis
    if 'left_knee_angle' in gait_data_df.columns:
        knee_jerk = np.mean(np.abs(np.diff(gait_data_df['left_knee_angle'].values, n=2)))
        if knee_jerk > 5:
            issues.append(f"⚡ JERKY MOVEMENT: Irregular knee motion detected")
    
    if not issues:
        issues.append("✅ NO SIGNIFICANT ISSUES DETECTED")
    
    return issues

# --- MAIN GAIT ANALYSIS FUNCTION ---
def analyze_gait(username, video_path, prosthetic_side, progress_callback):
    initialize_models()
    setup_database()
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    cap = cv2.VideoCapture(video_path)
    
    TARGET_WIDTH = 1280
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if total_frames == 0:
        raise ValueError("Video file is invalid or empty.")

    if original_width > TARGET_WIDTH:
        output_width = TARGET_WIDTH
        output_height = int(TARGET_WIDTH * (original_height / original_width))
    else:
        output_width, output_height = original_width, original_height
    
    # Add HUD height to video
    hud_height = 150
    video_height_with_hud = output_height + hud_height
        
    temp_output_filename = 'temp_annotated_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_filename, fourcc, fps, (output_width, video_height_with_hud))
    
    # Data collection
    gait_data = []
    left_steps = []
    right_steps = []
    stride_lengths = []
    step_widths = []
    toe_clearances = []
    com_positions = []
    
    # State tracking
    left_foot_down = False
    right_foot_down = False
    prev_left_ankle_y = None
    prev_right_ankle_y = None
    left_foot_pos = None
    right_foot_pos = None
    
    # Smoothing
    knee_history = {'left': deque(maxlen=7), 'right': deque(maxlen=7)}
    
    # Real-time issue detection
    current_issues = []
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        progress_callback(frame_count / total_frames)
        
        if original_width > TARGET_WIDTH:
            frame = cv2.resize(frame, (output_width, output_height))
        
        # Process with MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        
        # Create overlay
        overlay = frame.copy()
        
        # Clear current frame issues
        frame_issues = []
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Check landmark visibility
            visibility_threshold = 0.6
            landmarks_visible = all([
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > visibility_threshold,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > visibility_threshold,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > visibility_threshold,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > visibility_threshold
            ])
            
            if not landmarks_visible:
                frame_issues.append("⚠️ POOR VISIBILITY")
            
            # Extract key points
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * output_width,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * output_height]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * output_width,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * output_height]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * output_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * output_height]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * output_width,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * output_height]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * output_width,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * output_height]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * output_width,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * output_height]
            left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x * output_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y * output_height]
            right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x * output_width,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y * output_height]
            
            # Calculate angles
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            # Real-time angle check
            if abs(left_knee_angle - right_knee_angle) > 15:
                frame_issues.append("⚠️ HIGH ASYMMETRY")
            
            # Smooth angles
            knee_history['left'].append(left_knee_angle)
            knee_history['right'].append(right_knee_angle)
            left_knee_smooth = np.mean(knee_history['left'])
            right_knee_smooth = np.mean(knee_history['right'])
            
            # Pelvic angle
            pelvic_angle = calculate_angle([left_hip[0]-10, left_hip[1]], left_hip, right_hip)
            
            # Real-time pelvic check
            if abs(pelvic_angle - 90) > 15:
                frame_issues.append("⚠️ PELVIC TILT")
            
            # Step detection
            heel_threshold_y = 0.015 * output_height
            
            if prev_left_ankle_y is not None:
                left_ankle_movement = left_ankle[1] - prev_left_ankle_y
                
                # Left foot touchdown
                if left_ankle_movement > heel_threshold_y and not left_foot_down:
                    left_foot_down = True
                    left_steps.append(frame_count)
                    if right_foot_pos is not None:
                        stride_length = calculate_distance(left_ankle, right_foot_pos)
                        stride_lengths.append(stride_length)
                        step_width = abs(left_ankle[0] - right_foot_pos[0])
                        step_widths.append(step_width)
                        
                        # Real-time stride check
                        if len(stride_lengths) > 1:
                            stride_diff = abs(stride_lengths[-1] - stride_lengths[-2])
                            if stride_diff > 30:
                                frame_issues.append("⚡ IRREGULAR STRIDE")
                    
                    left_foot_pos = left_ankle.copy()
                    
                elif left_ankle_movement < -heel_threshold_y and left_foot_down:
                    left_foot_down = False
            
            if prev_right_ankle_y is not None:
                right_ankle_movement = right_ankle[1] - prev_right_ankle_y
                
                if right_ankle_movement > heel_threshold_y and not right_foot_down:
                    right_foot_down = True
                    right_steps.append(frame_count)
                    if left_foot_pos is not None:
                        stride_length = calculate_distance(right_ankle, left_foot_pos)
                        stride_lengths.append(stride_length)
                        step_width = abs(right_ankle[0] - left_foot_pos[0])
                        step_widths.append(step_width)
                    right_foot_pos = right_ankle.copy()
                    
                elif right_ankle_movement < -heel_threshold_y and right_foot_down:
                    right_foot_down = False
            
            # Toe clearance
            ground_level = max(left_heel[1], right_heel[1])
            if not left_foot_down:
                clearance = ground_level - left_ankle[1]
                toe_clearances.append(clearance)
                if clearance < 5:
                    frame_issues.append("🚨 LOW TOE CLEARANCE!")
            if not right_foot_down:
                clearance = ground_level - right_ankle[1]
                toe_clearances.append(clearance)
                if clearance < 5:
                    frame_issues.append("🚨 LOW TOE CLEARANCE!")
            
            prev_left_ankle_y = left_ankle[1]
            prev_right_ankle_y = right_ankle[1]
            
            # Center of mass
            com_x = (left_hip[0] + right_hip[0]) / 2
            com_y = (left_hip[1] + right_hip[1]) / 2
            com_positions.append([com_x, com_y])
            
            # Real-time balance check
            if len(com_positions) > 10:
                recent_com = np.array(com_positions[-10:])
                com_sway = np.std(recent_com[:, 0])
                if com_sway > 15:
                    frame_issues.append("⚠️ BALANCE ISSUE")
            
            # Store data
            gait_data.append({
                'frame': frame_count,
                'left_knee_angle': left_knee_angle,
                'right_knee_angle': right_knee_angle,
                'left_knee_smoothed': left_knee_smooth,
                'right_knee_smoothed': right_knee_smooth,
                'pelvic_tilt': pelvic_angle,
                'pelvic_tilt_smoothed': pelvic_angle,
                'left_foot_down': left_foot_down,
                'right_foot_down': right_foot_down,
                'com_x': com_x,
                'com_y': com_y
            })
            
            # --- ADVANCED VISUALIZATION ---
            
            # Draw skeleton base
            mp.solutions.drawing_utils.draw_landmarks(
                overlay, 
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=3),
                mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Highlight prosthetic leg with gradient effect
            prosthetic_color = (0, 255, 255)
            if prosthetic_side == "Left":
                prosthetic_points = [(left_hip, left_knee), (left_knee, left_ankle)]
            else:
                prosthetic_points = [(right_hip, right_knee), (right_knee, right_ankle)]
            
            for start, end in prosthetic_points:
                cv2.line(overlay, tuple(map(int, start)), tuple(map(int, end)), prosthetic_color, 5)
                # Add glow effect
                cv2.line(overlay, tuple(map(int, start)), tuple(map(int, end)), (255, 255, 255), 2)
            
            # Draw angle arcs with color coding
            def draw_enhanced_angle_arc(img, center, angle, label, radius=50):
                # Color code based on angle
                if 150 <= angle <= 180:
                    color = (0, 255, 0)  # Green - good extension
                elif 60 <= angle < 150:
                    color = (0, 255, 255)  # Yellow - flexion
                else:
                    color = (0, 0, 255)  # Red - abnormal
                
                # Draw arc
                cv2.ellipse(img, tuple(map(int, center)), (radius, radius), 0, 0, int(angle), color, 3)
                
                # Draw angle text with background
                text = f"{int(angle)}°"
                text_pos = (int(center[0]) + radius + 5, int(center[1]))
                
                # Text background
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (text_pos[0]-2, text_pos[1]-th-2), 
                            (text_pos[0]+tw+2, text_pos[1]+2), (0, 0, 0), -1)
                cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(img, label, (text_pos[0], text_pos[1]+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            draw_enhanced_angle_arc(overlay, left_knee, left_knee_smooth, "L", 50)
            draw_enhanced_angle_arc(overlay, right_knee, right_knee_smooth, "R", 50)
            
            # Draw step indicators with animation
            if left_foot_down:
                # Pulsing circle effect
                pulse_size = 20 + int(5 * np.sin(frame_count * 0.5))
                cv2.circle(overlay, tuple(map(int, left_ankle)), pulse_size, (0, 255, 0), -1)
                cv2.circle(overlay, tuple(map(int, left_ankle)), pulse_size+5, (0, 255, 0), 2)
                
                # Step label with background
                text = "L-CONTACT"
                text_pos = (int(left_ankle[0])-45, int(left_ankle[1])-25)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(overlay, (text_pos[0]-3, text_pos[1]-th-3), 
                            (text_pos[0]+tw+3, text_pos[1]+3), (0, 100, 0), -1)
                cv2.putText(overlay, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if right_foot_down:
                pulse_size = 20 + int(5 * np.sin(frame_count * 0.5))
                cv2.circle(overlay, tuple(map(int, right_ankle)), pulse_size, (0, 255, 0), -1)
                cv2.circle(overlay, tuple(map(int, right_ankle)), pulse_size+5, (0, 255, 0), 2)
                
                text = "R-CONTACT"
                text_pos = (int(right_ankle[0])-45, int(right_ankle[1])-25)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(overlay, (text_pos[0]-3, text_pos[1]-th-3), 
                            (text_pos[0]+tw+3, text_pos[1]+3), (0, 100, 0), -1)
                cv2.putText(overlay, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center of mass with trail
            cv2.circle(overlay, (int(com_x), int(com_y)), 10, (255, 0, 255), -1)
            cv2.circle(overlay, (int(com_x), int(com_y)), 12, (255, 255, 255), 2)
            
            # COM trail
            if len(com_positions) > 20:
                trail_points = np.array(com_positions[-20:], dtype=np.int32)
                for i in range(len(trail_points)-1):
                    alpha = i / len(trail_points)
                    cv2.line(overlay, tuple(trail_points[i]), tuple(trail_points[i+1]), 
                           (int(255*alpha), 0, int(255*alpha)), 2)
            
            # Draw pelvic line with level indicator
            cv2.line(overlay, tuple(map(int, left_hip)), tuple(map(int, right_hip)), (255, 255, 0), 4)
            
            # Pelvic level indicator
            pelvic_deviation = abs(pelvic_angle - 90)
            if pelvic_deviation < 5:
                pelvic_color = (0, 255, 0)
                pelvic_status = "LEVEL"
            elif pelvic_deviation < 10:
                pelvic_color = (0, 255, 255)
                pelvic_status = "SLIGHT TILT"
            else:
                pelvic_color = (0, 0, 255)
                pelvic_status = "TILTED"
            
            pelvic_text = f"PELVIS: {pelvic_status} ({pelvic_deviation:.1f}°)"
            text_pos = (int(com_x)-80, int(com_y)-25)
            (tw, th), _ = cv2.getTextSize(pelvic_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (text_pos[0]-3, text_pos[1]-th-3), 
                        (text_pos[0]+tw+3, text_pos[1]+3), (0, 0, 0), -1)
            cv2.putText(overlay, pelvic_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, pelvic_color, 2)
            
            # Draw stride length indicator
            if len(stride_lengths) > 0 and (left_foot_pos is not None and right_foot_pos is not None):
                cv2.line(overlay, tuple(map(int, left_foot_pos)), tuple(map(int, right_foot_pos)), 
                        (255, 165, 0), 2, cv2.LINE_AA)
                mid_point = ((left_foot_pos[0] + right_foot_pos[0])/2, 
                            (left_foot_pos[1] + right_foot_pos[1])/2)
                stride_text = f"{stride_lengths[-1]:.0f}px"
                cv2.putText(overlay, stride_text, tuple(map(int, mid_point)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        
        # Blend overlay
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        
        # --- ENHANCED HUD WITH REAL-TIME FEEDBACK ---
        hud = np.zeros((hud_height, output_width, 3), dtype=np.uint8)
        hud[:] = (30, 30, 30)
        
        # Top section - Progress bar
        progress_bar_width = int((frame_count / total_frames) * (output_width - 20))
        cv2.rectangle(hud, (10, 10), (output_width-10, 25), (60, 60, 60), -1)
        cv2.rectangle(hud, (10, 10), (10 + progress_bar_width, 25), (0, 255, 0), -1)
        progress_text = f"Frame: {frame_count}/{total_frames} ({frame_count*100//total_frames}%)"
        cv2.putText(hud, progress_text, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Stats section
        total_steps = len(left_steps) + len(right_steps)
        cv2.putText(hud, f"STEPS: {total_steps}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(hud, f"LEFT: {len(left_steps)}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        cv2.putText(hud, f"RIGHT: {len(right_steps)}", (10, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        # Prosthetic indicator
        cv2.putText(hud, f"PROSTHETIC: {prosthetic_side}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Real-time metrics
        if len(stride_lengths) > 0:
            avg_stride = np.mean(stride_lengths)
            cv2.putText(hud, f"AVG STRIDE: {avg_stride:.1f}px", (250, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if len(stride_lengths) > 1:
                stride_std = np.std(stride_lengths)
                consistency_color = (0, 255, 0) if stride_std < 15 else (0, 255, 255) if stride_std < 25 else (0, 0, 255)
                cv2.putText(hud, f"CONSISTENCY: {stride_std:.1f}", (250, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, consistency_color, 1)
        
        # Real-time cadence
        if total_steps > 0 and frame_count > 0:
            current_cadence = (total_steps / (frame_count / fps)) * 60
            cadence_color = (0, 255, 0) if 90 <= current_cadence <= 120 else (0, 255, 255) if current_cadence > 80 else (0, 0, 255)
            cv2.putText(hud, f"CADENCE: {current_cadence:.0f} spm", (250, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cadence_color, 1)
        
        # Real-time issue alerts (RIGHT SIDE)
        alert_y = 40
        cv2.putText(hud, "LIVE ALERTS:", (output_width - 250, alert_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        
        if frame_issues:
            for i, issue in enumerate(frame_issues[:4]):  # Show max 4 issues
                alert_y += 25
                # Determine color based on severity
                if "🚨" in issue:
                    color = (0, 0, 255)  # Red - Critical
                elif "⚠️" in issue:
                    color = (0, 165, 255)  # Orange - Warning
                else:
                    color = (0, 255, 255)  # Yellow - Caution
                
                cv2.putText(hud, issue, (output_width - 250, alert_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(hud, "✅ NO ISSUES", (output_width - 250, alert_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Combine HUD with frame
        frame_with_hud = np.vstack([hud, frame])
        
        out.write(frame_with_hud)
    
    cap.release()
    out.release()
    pose.close()

    # Convert to H.264
    final_output_filename = 'annotated_video_final.mp4'
    try:
        (
            ffmpeg
            .input(temp_output_filename)
            .output(final_output_filename, vcodec='libx264', pix_fmt='yuv420p', crf=23)
            .run(overwrite_output=True, quiet=True)
        )
    except ffmpeg.Error as e:
        print("FFmpeg Error:", e.stderr.decode() if e.stderr else "Unknown error")
        final_output_filename = temp_output_filename
    
    if os.path.exists(temp_output_filename) and final_output_filename != temp_output_filename:
        os.remove(temp_output_filename)

    if not gait_data:
        return None, "Analysis failed - no pose data detected.", None, None, None, None

    # --- DATA ANALYSIS ---
    df = pd.DataFrame(gait_data)
    
    # Apply smoothing
    window_length = min(11, len(df) if len(df) % 2 == 1 else len(df) - 1)
    if window_length >= 5:
        df['left_knee_smoothed'] = savgol_filter(df['left_knee_angle'], window_length, 3)
        df['right_knee_smoothed'] = savgol_filter(df['right_knee_angle'], window_length, 3)
        df['pelvic_tilt_smoothed'] = savgol_filter(df['pelvic_tilt'], window_length, 3)

    # Calculate metrics
    total_steps = len(left_steps) + len(right_steps)
    video_duration_minutes = total_frames / (fps * 60)
    cadence = total_steps / video_duration_minutes if video_duration_minutes > 0 else 0
    
    avg_left_knee = df['left_knee_smoothed'].mean()
    avg_right_knee = df['right_knee_smoothed'].mean()
    asymmetry = abs(avg_left_knee - avg_right_knee)
    
    avg_pelvic_tilt = df['pelvic_tilt_smoothed'].mean()
    pelvic_tilt_range = df['pelvic_tilt_smoothed'].max() - df['pelvic_tilt_smoothed'].min()
    
    avg_stride_length_px = np.mean(stride_lengths) if stride_lengths else 0
    stride_variability = np.std(stride_lengths) if len(stride_lengths) > 1 else 0
    
    avg_step_width = np.mean(step_widths) if step_widths else 0
    toe_clearance_min = abs(np.min(toe_clearances)) if toe_clearances else 0
    
    # Gait smoothness (jerk metric)
    knee_jerk_left = np.mean(np.abs(np.diff(df['left_knee_smoothed'].values, n=2)))
    knee_jerk_right = np.mean(np.abs(np.diff(df['right_knee_smoothed'].values, n=2)))
    gait_smoothness = (knee_jerk_left + knee_jerk_right) / 2
    
    # Balance score (COM stability)
    if len(com_positions) > 10:
        com_array = np.array(com_positions)
        com_sway = np.std(com_array[:, 0])
        balance_score = max(0, 100 - com_sway * 2)  # Higher is better
    else:
        balance_score = 50

    # Deep Learning fall risk assessment
    fall_risk_category, final_risk_score, pattern_type, anomaly_detected, dl_confidence = assess_fall_risk_dl(
        df, asymmetry, stride_variability, pelvic_tilt_range, cadence, avg_step_width, toe_clearance_min
    )
    
    # Detect specific issues
    detected_issues_list = detect_gait_issues(
        df, asymmetry, stride_variability, cadence, toe_clearance_min, pelvic_tilt_range, avg_step_width
    )
    detected_issues_str = " | ".join(detected_issues_list)

    # Create enhanced visualizations
    knee_fig = go.Figure()
    knee_fig.add_trace(go.Scatter(
        x=df['frame'], y=df['left_knee_smoothed'], 
        mode='lines', name='Left Knee',
        line=dict(color='rgb(0,176,246)', width=2.5),
        hovertemplate='Frame: %{x}<br>Angle: %{y:.1f}°<extra></extra>'
    ))
    knee_fig.add_trace(go.Scatter(
        x=df['frame'], y=df['right_knee_smoothed'], 
        mode='lines', name='Right Knee',
        line=dict(color='rgb(231,107,243)', width=2.5),
        hovertemplate='Frame: %{x}<br>Angle: %{y:.1f}°<extra></extra>'
    ))
    
    # Add step markers
    for step_frame in left_steps:
        knee_fig.add_vline(x=step_frame, line_dash="dot", line_color="rgba(0,176,246,0.3)", 
                          annotation_text="L", annotation_position="top")
    for step_frame in right_steps:
        knee_fig.add_vline(x=step_frame, line_dash="dot", line_color="rgba(231,107,243,0.3)", 
                          annotation_text="R", annotation_position="top")
    
    knee_fig.update_layout(
        title=f"Knee Angle Analysis - Pattern: {pattern_type}",
        xaxis_title="Frame Number",
        yaxis_title="Knee Angle (degrees)",
        hovermode='x unified',
        template='plotly_white',
        height=450
    )

    pelvic_fig = go.Figure()
    pelvic_fig.add_trace(go.Scatter(
        x=df['frame'], y=df['pelvic_tilt_smoothed'], 
        mode='lines', name='Pelvic Tilt',
        line=dict(color='rgb(255,193,7)', width=2.5),
        fill='tozeroy', fillcolor='rgba(255,193,7,0.2)',
        hovertemplate='Frame: %{x}<br>Tilt: %{y:.1f}°<extra></extra>'
    ))
    
    # Add reference line for ideal
    pelvic_fig.add_hline(y=90, line_dash="dash", line_color="green", 
                        annotation_text="Ideal Level", annotation_position="right")
    
    pelvic_fig.update_layout(
        title="Pelvic Stability Analysis",
        xaxis_title="Frame Number",
        yaxis_title="Pelvic Angle (degrees)",
        hovermode='x unified',
        template='plotly_white',
        height=450
    )

    # Stride variability plot
    metrics_fig = go.Figure()
    
    if stride_lengths:
        stride_frames = [left_steps[i] if i < len(left_steps) else right_steps[i-len(left_steps)] 
                        for i in range(len(stride_lengths))]
        metrics_fig.add_trace(go.Scatter(
            x=stride_frames, y=stride_lengths,
            mode='lines+markers', name='Stride Length',
            line=dict(color='rgb(46,204,113)', width=2),
            marker=dict(size=8, color='rgb(46,204,113)'),
            hovertemplate='Frame: %{x}<br>Stride: %{y:.1f}px<extra></extra>'
        ))
        
        # Add mean line
        metrics_fig.add_hline(y=avg_stride_length_px, line_dash="dash", line_color="blue",
                             annotation_text=f"Mean: {avg_stride_length_px:.1f}px")
    
    metrics_fig.update_layout(
        title=f"Stride Pattern - Variability: {stride_variability:.1f}px",
        xaxis_title="Frame Number",
        yaxis_title="Stride Length (pixels)",
        hovermode='x unified',
        template='plotly_white',
        height=450
    )

    # Summary data
    summary_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'video_name': os.path.basename(video_path),
        'total_steps': total_steps,
        'cadence': round(cadence, 2),
        'avg_left_knee_angle': round(avg_left_knee, 2),
        'avg_right_knee_angle': round(avg_right_knee, 2),
        'asymmetry': round(asymmetry, 2),
        'avg_pelvic_tilt': round(avg_pelvic_tilt, 2),
        'pelvic_tilt_range': round(pelvic_tilt_range, 2),
        'avg_stride_length_px': round(avg_stride_length_px, 2),
        'stride_variability': round(stride_variability, 2),
        'avg_step_width': round(avg_step_width, 2),
        'toe_clearance_min': round(toe_clearance_min, 2),
        'gait_smoothness': round(gait_smoothness, 2),
        'balance_score': round(balance_score, 2),
        'fall_risk': fall_risk_category,
        'final_risk_score': final_risk_score,
        'pattern_type': pattern_type,
        'dl_confidence': round(dl_confidence * 100, 1),
        'anomaly_detected': anomaly_detected,
        'detected_issues': detected_issues_str
    }
    
    save_session_to_db(username, summary_data)

    return final_output_filename, summary_data, df, knee_fig, pelvic_fig, metrics_fig
