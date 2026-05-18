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

import math

def get_line_props(line):
    x1, y1, x2, y2 = line
    length = math.hypot(x2 - x1, y2 - y1)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
    
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    denom = math.hypot(A, B)
    rho = abs(C) / denom if denom != 0 else 0
    return length, angle, rho

def merge_collinear_lines(lines, max_angle_diff=8, max_rho_diff=15, max_gap=150):
    if len(lines) == 0:
        return []
        
    segments = [line[0] for line in lines]
    merged = []
    used = [False] * len(segments)
    
    for i in range(len(segments)):
        if used[i]:
            continue
            
        group = [segments[i]]
        used[i] = True
        _, ang_i, rho_i = get_line_props(segments[i])
        
        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
                
            _, ang_j, rho_j = get_line_props(segments[j])
            
            ang_diff = abs(ang_i - ang_j)
            ang_diff = min(ang_diff, 180 - ang_diff)
            rho_diff = abs(rho_i - rho_j)
            
            if ang_diff < max_angle_diff and rho_diff < max_rho_diff:
                group.append(segments[j])
                used[j] = True
                
        x1, y1, x2, y2 = group[0]
        is_horizontal = abs(x2 - x1) > abs(y2 - y1)
        
        if is_horizontal:
            group.sort(key=lambda s: min(s[0], s[2]))
            current = group[0]
            for next_seg in group[1:]:
                c_left = min(current[0], current[2])
                c_right = max(current[0], current[2])
                n_left = min(next_seg[0], next_seg[2])
                n_right = max(next_seg[0], next_seg[2])
                
                if n_left - c_right < max_gap:
                    current = [min(c_left, n_left), int((current[1]+next_seg[1])/2),
                               max(c_right, n_right), int((current[3]+next_seg[3])/2)]
                else:
                    merged.append(current)
                    current = next_seg
            merged.append(current)
        else:
            group.sort(key=lambda s: min(s[1], s[3]))
            current = group[0]
            for next_seg in group[1:]:
                c_top = min(current[1], current[3])
                c_bot = max(current[1], current[3])
                n_top = min(next_seg[1], next_seg[3])
                n_bot = max(next_seg[1], next_seg[3])
                
                if n_top - c_bot < max_gap:
                    current = [int((current[0]+next_seg[0])/2), min(c_top, n_top),
                               int((current[2]+next_seg[2])/2), max(c_bot, n_bot)]
                else:
                    merged.append(current)
                    current = next_seg
            merged.append(current)
            
    return merged

def merge_parallel_lines(lines, orientation='horizontal', max_dist=25):
    if len(lines) == 0:
        return []
        
    merged = []
    used = [False] * len(lines)
    
    if orientation == 'horizontal':
        lines.sort(key=lambda l: (l[1] + l[3]) / 2)
    else:
        lines.sort(key=lambda l: (l[0] + l[2]) / 2)
        
    for i in range(len(lines)):
        if used[i]:
            continue
            
        group = [lines[i]]
        used[i] = True
        
        coord_i = (lines[i][1] + lines[i][3])/2 if orientation == 'horizontal' else (lines[i][0] + lines[i][2])/2
        
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
                
            coord_j = (lines[j][1] + lines[j][3])/2 if orientation == 'horizontal' else (lines[j][0] + lines[j][2])/2
            
            if abs(coord_i - coord_j) < max_dist:
                if orientation == 'horizontal':
                    left_i, right_i = min(lines[i][0], lines[i][2]), max(lines[i][0], lines[i][2])
                    left_j, right_j = min(lines[j][0], lines[j][2]), max(lines[j][0], lines[j][2])
                    overlap = min(right_i, right_j) - max(left_i, left_j)
                    if overlap > -50:
                        group.append(lines[j])
                        used[j] = True
                else:
                    top_i, bot_i = min(lines[i][1], lines[i][3]), max(lines[i][1], lines[i][3])
                    top_j, bot_j = min(lines[j][1], lines[j][3]), max(lines[j][1], lines[j][3])
                    overlap = min(bot_i, bot_j) - max(top_i, top_j)
                    if overlap > -50:
                        group.append(lines[j])
                        used[j] = True
                        
        if len(group) == 1:
            merged.append(group[0])
        else:
            if orientation == 'horizontal':
                min_x = min([min(l[0], l[2]) for l in group])
                max_x = max([max(l[0], l[2]) for l in group])
                y_avg = int(np.mean([l[1] + l[3] for l in group]) / 2)
                merged.append([min_x, y_avg, max_x, y_avg])
            else:
                min_y = min([min(l[1], l[3]) for l in group])
                max_y = max([max(l[1], l[3]) for l in group])
                x_avg = int(np.mean([l[0] + l[2] for l in group]) / 2)
                merged.append([x_avg, min_y, x_avg, max_y])
                
    return merged

def get_projection_chart(profile, smoothed, peaks, threshold, orientation):
    """Generate Matplotlib projection plot in-memory and return base64 string"""
    plt.figure(figsize=(8, 3))
    plt.plot(profile, label='Optical Density Profile', alpha=0.4, color='#94a3b8')
    plt.plot(smoothed, label='Smoothed Light Energy', color='#818cf8', linewidth=2)
    
    # Check if peak coordinate values exist in profile range to draw safely
    valid_peaks = [p for p in peaks if 0 <= p < len(smoothed)]
    if valid_peaks:
        plt.plot(valid_peaks, smoothed[valid_peaks], "ro", label=f'Rods (Count={len(peaks)})', markersize=6)
    
    plt.title(f"1D Optical Intensity Profile & Peak Mapping ({orientation.capitalize()})", fontsize=10, color='#f1f5f9', fontweight='bold')
    plt.xlabel("Pixel Coordinate", fontsize=8, color='#94a3b8')
    plt.ylabel("Average Brightness", fontsize=8, color='#94a3b8')
    
    fig = plt.gcf()
    ax = plt.gca()
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
        
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    
    legend = plt.legend(facecolor='#1e293b', edgecolor='#334155', fontsize=8)
    for text in legend.get_texts():
        text.set_color('#f1f5f9')
        
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', dpi=120)
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{chart_base64}"

def run_pipeline(img_path, orientation, peak_ratio, min_spacing, smoothing_win, kernel_sz):
    """Advanced line segment tracking and parallel merging pipeline"""
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. White Top-Hat Morphological Filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_sz, kernel_sz))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # 2. Canny Edge Extraction
    edges = cv2.Canny(tophat, 50, 150)
    
    # 3. Hough Lines Extraction
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=50, maxLineGap=10)
    
    # Filter raw segments by dominant orientation angle
    filtered_lines = []
    if raw_lines is not None:
        for line in raw_lines:
            _, angle, _ = get_line_props(line[0])
            if orientation == 'horizontal':
                if angle < 15 or angle > 165:
                    filtered_lines.append(line)
            else:
                if abs(angle - 90) < 15:
                    filtered_lines.append(line)
                    
    # 4. Collinear Merging (Connect endpoints along same line axis)
    collinear_merged = merge_collinear_lines(filtered_lines, max_angle_diff=8, max_rho_diff=15, max_gap=150)
    
    # 5. Parallel Line Suppression (Thickness suppression)
    # Uses 'min_spacing' parameter dynamically from slider to control NMS distance
    final_rods = merge_parallel_lines(collinear_merged, orientation=orientation, max_dist=min_spacing)
    
    # Filter out very short segments (noise)
    min_rod_len = 150 # pixels
    final_rods = [l for l in final_rods if math.hypot(l[2]-l[0], l[3]-l[1]) >= min_rod_len]
    
    # Draw centerline detections strictly between endpoints
    detected_img = img.copy()
    peaks = []
    
    for i, line in enumerate(final_rods):
        x1, y1, x2, y2 = line
        # Draw the line strictly on the rod
        cv2.line(detected_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # Highlight boundaries with green dots
        cv2.circle(detected_img, (x1, y1), 6, (0, 255, 0), -1)
        cv2.circle(detected_img, (x2, y2), 6, (0, 255, 0), -1)
        # Add labels
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        cv2.putText(detected_img, str(i + 1), (mid_x, mid_y - 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Calculate peak coordinate perpendicular to direction for graph mapping
        coord = int((y1 + y2) / 2) if orientation == 'horizontal' else int((x1 + x2) / 2)
        peaks.append(coord)
        
    # Generate optical density profile for diagnostic visual chart
    axis = 1 if orientation == 'horizontal' else 0
    profile = np.mean(tophat, axis=axis)
    if smoothing_win % 2 == 0:
        smoothing_win += 1
    smoothed = np.convolve(profile, np.ones(smoothing_win)/smoothing_win, mode='same')
    
    # Convert results to Base64
    orig_base64 = encode_image_base64(detected_img)
    tophat_base64 = encode_image_base64(tophat)
    chart_base64 = get_projection_chart(profile, smoothed, peaks, 0, orientation)
    
    return {
        'count': len(final_rods),
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
