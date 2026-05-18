import os
import cv2
import numpy as np
import base64
import io
import matplotlib
matplotlib.use('Agg') # Thread-safe headless mode for web servers
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = os.path.join('d:\\stressing_metal_strings_counter', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB limit

def encode_image_base64(img):
    """Convert OpenCV image to base64 string for direct HTML embedding"""
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def get_projection_chart(profile, smoothed, peaks, threshold, orientation):
    """Generate Matplotlib projection plot in-memory and return base64 string"""
    plt.figure(figsize=(8, 3))
    plt.plot(profile, label='Raw Intensity', alpha=0.4, color='#94a3b8')
    plt.plot(smoothed, label='Smoothed Signal', color='#38bdf8', linewidth=2)
    plt.plot(peaks, smoothed[peaks], "ro", label=f'Rods (Count={len(peaks)})', markersize=6)
    plt.axhline(y=threshold, color='#f87171', linestyle='--', label='Threshold Line', alpha=0.8)
    
    plt.title(f"1D Intensity Projection ({orientation.capitalize()} Sum)", fontsize=10, color='#f1f5f9', fontweight='bold')
    plt.xlabel("Pixel Coordinate", fontsize=8, color='#94a3b8')
    plt.ylabel("Average Brightness", fontsize=8, color='#94a3b8')
    
    # Style chart for dark mode integration
    fig = plt.gcf()
    ax = plt.gca()
    fig.patch.set_facecolor('#0f172a') # matches website card background
    ax.set_facecolor('#0f172a')
    
    # Tick colors
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
        
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    
    legend = plt.legend(facecolor='#1e293b', edgecolor='#334155', fontsize=8)
    for text in legend.get_texts():
        text.set_color('#f1f5f9')
        
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', dpi=120)
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{chart_base64}"

def run_pipeline(img_path, orientation, peak_ratio, min_spacing, smoothing_win, kernel_sz):
    """Core image processing pipeline matching rpi_rod_counter"""
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. White Top-Hat Morphological Filter (Uneven lighting cleaner)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_sz, kernel_sz))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # 2. 1D Projection
    axis = 1 if orientation == 'horizontal' else 0
    profile = np.mean(tophat, axis=axis)
    
    # 3. 1D Signal Smoothing
    if smoothing_win % 2 == 0:
        smoothing_win += 1 # must be odd for standard moving window
    smoothed = np.convolve(profile, np.ones(smoothing_win)/smoothing_win, mode='same')
    
    # 4. Peak Detection (Local maxima with distance threshold)
    peaks = []
    max_val = np.max(smoothed)
    threshold_val = max_val * (peak_ratio / 100.0) if max_val > 5 else 255
    
    for i in range(min_spacing, len(smoothed) - min_spacing):
        if smoothed[i] > threshold_val:
            neighborhood = smoothed[i - min_spacing : i + min_spacing + 1]
            if smoothed[i] == np.max(neighborhood):
                if len(peaks) == 0 or (i - peaks[-1]) >= min_spacing:
                    peaks.append(i)
                    
    # Draw centerline detections on original
    detected_img = img.copy()
    for idx in peaks:
        if orientation == 'horizontal':
            cv2.line(detected_img, (0, idx), (img.shape[1], idx), (0, 0, 255), 3)
        else:
            cv2.line(detected_img, (idx, 0), (idx, img.shape[0]), (0, 0, 255), 3)
            
    # Generate visualization representations
    orig_base64 = encode_image_base64(detected_img)
    tophat_base64 = encode_image_base64(tophat)
    chart_base64 = get_projection_chart(profile, smoothed, peaks, threshold_val, orientation)
    
    return {
        'count': len(peaks),
        'original_img': orig_base64,
        'tophat_img': tophat_base64,
        'chart_img': chart_base64
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Default settings for initial load
        results = run_pipeline(
            img_path=filepath,
            orientation='horizontal',
            peak_ratio=18,
            min_spacing=15,
            smoothing_win=11,
            kernel_sz=15
        )
        
        if results is None:
            return jsonify({'error': 'Failed to process image'}), 500
            
        return jsonify({
            'success': True,
            'filename': filename,
            'results': results
        })

@app.route('/process', methods=['POST'])
def process_parameters():
    data = request.json
    filename = data.get('filename')
    orientation = data.get('orientation', 'horizontal')
    threshold = int(data.get('threshold', 18))
    spacing = int(data.get('spacing', 15))
    smoothing = int(data.get('smoothing', 11))
    kernel = int(data.get('kernel', 15))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Image file no longer exists'}), 404
        
    results = run_pipeline(
        img_path=filepath,
        orientation=orientation,
        peak_ratio=threshold,
        min_spacing=spacing,
        smoothing_win=smoothing,
        kernel_sz=kernel
    )
    
    if results is None:
        return jsonify({'error': 'Processing failed'}), 500
        
    return jsonify({
        'success': True,
        'results': results
    })

if __name__ == '__main__':
    print("--------------------------------------------------")
    print("Stressing Metal Strings Counter - Windows Server")
    print("Open http://127.0.0.1:5000 in your browser to run.")
    print("--------------------------------------------------")
    app.run(debug=True, host='127.0.0.1', port=5000)
