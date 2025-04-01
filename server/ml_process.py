import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from scipy.fftpack import fft
from flask import jsonify
from scipy.interpolate import interp1d

def extract_xgboost_features(sensor_data):
    """ Extract statistical, frequency, and time-domain features """
   # Basic Stats
    basic_stats = np.array([
        np.mean(sensor_data),
        np.std(sensor_data),
        np.min(sensor_data),
        np.max(sensor_data),
        skew(sensor_data),
        kurtosis(sensor_data)
    ])

    # Frequency Features (FFT)
    fft_values = np.abs(fft(sensor_data))[:len(sensor_data) // 2]
    dom_freq = np.argmax(fft_values, axis=0)  # Most dominant frequency
    spectral_entropy = -np.sum((fft_values / np.sum(fft_values)) * np.log2(fft_values + 1e-6), axis=0)  # Avoid log(0)

    # Time-Domain Features (RMS & Peaks)
    rms = np.sqrt(np.mean(sensor_data ** 2))  # Energy level
    peaks, _ = find_peaks(sensor_data, height=np.mean(sensor_data) + np.std(sensor_data))  # Peak detection

    # Combine All Features
    features = np.concatenate([basic_stats, [dom_freq, spectral_entropy, rms, len(peaks)]])

    return features

def extract_all_features(gyro_data, accel_data, timestamps):
    """ Extract features for gyro and accel separately and combine them """

    # Extract gyro features
    gyro_features = extract_xgboost_features(gyro_data)

    # Extract accel features
    accel_features = extract_xgboost_features(accel_data)

    # Compute time differences (for event duration)
    time_diffs = np.diff(timestamps)
    avg_time_gap = np.mean(time_diffs) if len(time_diffs) > 0 else 0

     # Cross-Signal Features
    corr = np.corrcoef(gyro_data, accel_data)[0, 1]  # Pearson correlation
    mean_diff = np.mean(np.abs(gyro_data - accel_data))  # Mean absolute difference
    energy_ratio = np.sqrt(np.mean(gyro_data ** 2)) / np.sqrt(np.mean(accel_data ** 2))  # Energy ratio

    # Merge Features
    combined_features = np.concatenate([gyro_features, accel_features, [avg_time_gap, corr, mean_diff, energy_ratio]])

    return combined_features

def preprocess_csv_knn(file, target_fs=50, duration=2.0):
    """Load and resample sensor data to a fixed frequency."""
    df = pd.read_csv(file)

    # Ensure required columns exist
    expected_cols = {'timestamp', 'gyroMag', 'accelMag'}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {expected_cols - set(df.columns)}")

    # Convert timestamps to relative time
    df['timestamp'] -= df['timestamp'].min()
    total_time = df['timestamp'].max()

    # Handle duplicate timestamps by averaging values
    df = df.groupby('timestamp', as_index=False).agg({'gyroMag': 'mean', 'accelMag': 'mean'})

    # Normalize timestamps to fit within `duration`
    if total_time > 0:
        df['timestamp'] *= (duration / total_time)

    # Define new evenly spaced timestamps (0 to duration, at target_fs Hz)
    new_time = np.linspace(0, duration, int(target_fs * duration))

    # Interpolation to resample data
    interp_gyro = interp1d(df['timestamp'], df['gyroMag'], kind='linear', fill_value='extrapolate')
    interp_accel = interp1d(df['timestamp'], df['accelMag'], kind='linear', fill_value='extrapolate')

    # Get resampled values
    new_gyro = interp_gyro(new_time)
    new_accel = interp_accel(new_time)

    return new_gyro, new_accel, new_time

def predict(request, app):
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
             # Get injected dependencies
        model = app.config['model']
        scaler = app.config['scaler']
        label_encoder = app.config['label_encoder']
        
        # Convert to numpy and apply scaling
        gyro, accel, timestamp = preprocess_csv_knn(file)
        features = extract_all_features(gyro, accel, timestamp)
        X_features = features.reshape(1, -1) if features.ndim == 1 else features
        X_features = scaler.transform(X_features)  # Normalize

        # Run inference
        predictions = model.predict(X_features)
        predicted_labels = label_encoder.inverse_transform(predictions)  # Convert back to original labels

        return jsonify({"predictions": predicted_labels.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500