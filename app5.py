# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import wfdb
from tensorflow.keras.models import load_model
from scipy.signal import butter, filtfilt
import neurokit2 as nk
import plotly.graph_objects as go
import json
import os
app = Flask(__name__)

# Constants
DATASET_PATH = "mit-bih-arrhythmia-database-1.0.0/"
MODEL_PATH = os.path.join('model', 'ecg_arrhythmia_detector_20250331_165229.h5')

CLASSES = {
    0: {'id': 'N', 'name': 'Normal', 'weight': 1, 'color': 'green'},
    1: {'id': 'S', 'name': 'SVT', 'weight': 100, 'color': 'orange'},
    2: {'id': 'AF', 'name': 'Atrial Fibrillation', 'weight': 150, 'color': 'red'},
    3: {'id': 'VF', 'name': 'Ventricular Fibrillation', 'weight': 200, 'color': 'purple'},
    4: {'id': 'VT', 'name': 'Ventricular Tachycardia', 'weight': 170, 'color': 'darkred'},
    5: {'id': 'B', 'name': 'Heart Block', 'weight': 120, 'color': 'blue'},
    6: {'id': 'F', 'name': 'Fusion', 'weight': 80, 'color': 'brown'}
}

# Load model
try:
    model = load_model(MODEL_PATH)
except:
    model = None
    print("Warning: Model not loaded")

def load_ecg_sample(record_num="100"):
    try:
        record = wfdb.rdrecord(f"{DATASET_PATH}{record_num}")
        annotation = wfdb.rdann(f"{DATASET_PATH}{record_num}", 'atr')
        return record.p_signal[:, 0], record.fs, annotation.symbol, annotation.sample
    except Exception as e:
        print(f"Error loading record {record_num}: {str(e)}")
        return None, None, None, None

def butterworth_filter(signal, cutoff=50, fs=360, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def preprocess_ecg(ecg_signal, target_length=180):
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
    if len(ecg_signal) > target_length:
        ecg_signal = ecg_signal[:target_length]
    else:
        ecg_signal = np.pad(ecg_signal, (0, target_length - len(ecg_signal)))
    return np.expand_dims(np.expand_dims(ecg_signal, axis=-1), axis=0)

def handle_model_output(predictions):
    try:
        if isinstance(predictions, list) and len(predictions) > 0:
            return np.array(predictions[0][0][:7])
        return np.array(predictions[0][0][:7])
    except Exception as e:
        print(f"Error processing output: {str(e)}")
        return None

def create_ecg_plot(ecg_signal, r_peaks, pred_class_id, fs=360, samples_to_show=2000):
    fig = go.Figure()
    
    # Add ECG trace
    fig.add_trace(go.Scatter(
        x=list(range(samples_to_show)),
        y=ecg_signal[:samples_to_show].tolist(),
        mode='lines',
        line=dict(color='lightgray', width=1),
        name='ECG Signal'
    ))
    
    # Highlight beats
    if len(r_peaks) > 0:
        visible_r_peaks = r_peaks[r_peaks < samples_to_show]
        class_info = next(v for k, v in CLASSES.items() if v['id'] == pred_class_id)
        
        fig.add_trace(go.Scatter(
            x=visible_r_peaks.tolist(),
            y=ecg_signal[visible_r_peaks].tolist(),
            mode='markers',
            marker=dict(color=class_info['color'], size=8),
            name=f'{pred_class_id} beats'
        ))
    
    fig.update_layout(
        title=f'ECG with {pred_class_id} Beat Pattern',
        xaxis_title='Samples',
        yaxis_title='Amplitude',
        showlegend=True,
        template='plotly_white'
    )
    
    return json.loads(fig.to_json())

@app.route('/')
def index():
    record_options = [str(i) for i in range(100, 235) if i not in [102, 104, 107, 217]]
    return render_template('index.html', record_options=record_options, classes=CLASSES)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    record_num = data.get('record_num', '100')
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    ecg_signal, fs, _, _ = load_ecg_sample(record_num)
    if ecg_signal is None:
        return jsonify({'error': 'Failed to load ECG data'}), 400
    
    try:
        # Preprocess
        ecg_signal = butterworth_filter(ecg_signal, fs=fs)
        processed_ecg = preprocess_ecg(ecg_signal)
        
        # Detect R-peaks
        r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=fs)[1]['ECG_R_Peaks']
        
        # Predict
        predictions = model.predict(processed_ecg, verbose=0)
        class_probs = handle_model_output(predictions)
        
        if class_probs is None:
            return jsonify({'error': 'Invalid model output'}), 500
        
        # Get results
        pred_idx = np.argmax(class_probs)
        confidence = class_probs[pred_idx] * 100
        pred_class = CLASSES[pred_idx]
        
        # Create plot
        plot_data = create_ecg_plot(ecg_signal, r_peaks, pred_class['id'], fs)
        
        return jsonify({
            'prediction': pred_class['name'],
            'class_id': pred_class['id'],
            'confidence': float(confidence),
            'priority': pred_class['weight'],
            'plot_data': plot_data,
            'heart_rate': float(60 / np.mean(np.diff(r_peaks)/fs)) if len(r_peaks) > 1 else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)