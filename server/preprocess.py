import numpy as np
import tensorflow as tf
import pandas as pd
import scipy
import cv2
from io import BytesIO
from scipy.interpolate import interp1d
from flask import send_file, jsonify

global_min_gyro = 0
global_min_accel = 0
global_max_gyro = 30
global_max_accel = 80

def preprocess_csv(file, target_fs=50, duration=2.0, gap_threshold=30):
    # Load CSV (Assume columns: timestamp, gyro_mag, accel_mag)
    df = pd.read_csv(file)

    # Ensure required columns exist
    expected_cols = {'timestamp', 'gyroMag', 'accelMag'}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {expected_cols - set(df.columns)}")

    # Convert timestamps to relative time (start from 0)
    df['timestamp'] -= df['timestamp'].min()
    total_time = df['timestamp'].max()

    # Handle duplicate timestamps by averaging values
    df = df.groupby('timestamp', as_index=False).agg({'gyroMag': 'mean', 'accelMag': 'mean'})

    # Normalize timestamps to the desired duration
    if total_time > 0:
        df['timestamp'] *= (duration / total_time)

    # Define new evenly spaced timestamps (0 to duration, at target_fs Hz)
    new_time = np.linspace(0, duration, int(target_fs * duration))

    # Interpolation (Ensures uniform time steps)
    interp_gyro = interp1d(df['timestamp'], df['gyroMag'], kind='linear', fill_value='extrapolate')
    interp_accel = interp1d(df['timestamp'], df['accelMag'], kind='linear', fill_value='extrapolate')

    # Get resampled values
    new_gyro = interp_gyro(new_time)
    new_accel = interp_accel(new_time)

    # Compute Min-Max Scaling dynamically
    if global_max_gyro > global_min_gyro:
        new_gyro = (new_gyro - global_min_gyro) / (global_max_gyro - global_min_gyro)
    else:
        new_gyro = np.zeros_like(new_gyro)  # Avoid division by zero

    if global_max_accel > global_min_accel:
        new_accel = (new_accel - global_min_accel) / (global_max_accel - global_min_accel)
    else:
        new_accel = np.zeros_like(new_accel)

    return new_time, new_gyro, new_accel

def generate_spectrogram(signal, fs=50, WR=5, duration=2, overlap_percent=50):
    """Compute and return the spectrogram of a given signal."""
    # Compute total samples N based on sampling frequency and duration
    N = int(fs * duration)  # N = fs * 2 for your 2-second data

    # Compute window size based on WR and total samples N
    W_size = N // WR  # W_size = N / WR

    # Compute the overlap in samples
    noverlap = int(W_size * (overlap_percent / 100))

    # Use Hamming window explicitly (to match the paper)
    window = scipy.signal.get_window('hamming', W_size)
    # Compute spectrogram with Hamming window
    f, t, Sxx = scipy.signal.spectrogram(signal, fs=fs, window=window, nperseg=W_size, noverlap=noverlap)

    # Apply log transform to reduce dynamic range
    Sxx = np.log1p(Sxx)

    return f, t, Sxx


def generate_spectrogram_image(Sxx):
    """Generate spectrogram using OpenCV and return as in-memory image."""
    Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min())  # Normalize to [0,1]
    Sxx = (Sxx * 255).astype(np.uint8)  # Convert to uint8 grayscale image

    # Apply colormap (Jet for a more visually appealing spectrogram)
    spectrogram_img = cv2.applyColorMap(Sxx, cv2.COLORMAP_JET)

    # Resize to 224x224 for MobileNet compatibility
    spectrogram_img = cv2.resize(spectrogram_img, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Encode as PNG in memory
    _, buffer = cv2.imencode('.jpg', spectrogram_img)
    img_io = BytesIO(buffer)

    return img_io

def preprocess_api(request):
    if 'file' not in request.files:
      return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        new_time, new_gyro, new_accel = preprocess_csv(file)
        return jsonify({"timestamp": new_time, "gyroMag": new_gyro, "accelMag": new_accel})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def spectrogram_img(request):
    if 'file' not in request.files:
        return {"error": "No file uploaded"}, 400
    
    file = request.files['file']
    df = np.loadtxt(file, delimiter=',')  # Assuming a CSV with a single column of signal data

    if df.ndim != 1:
        return {"error": "Invalid file format, expected single-column CSV"}, 400

    f, t, Sxx = generate_spectrogram(df)
    img_io = generate_spectrogram_image(Sxx)
    return send_file(img_io, mimetype='image/jpg')