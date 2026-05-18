"""
Real-time Stressing Metal Strings/Rods Counter for Raspberry Pi 4.
Optimized for ultra-low latency, robust background rejection, and industrial physical output.
"""

import cv2
import numpy as np
import time
import os
import csv

# Try importing RPi.GPIO to support physical outputs on Raspberry Pi
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("RPi.GPIO not found. Running in simulation mode (GPIO outputs disabled).")

# ==================== CONFIGURATION ====================
CAMERA_INDEX = 0          # 0 for default Pi camera / USB webcam
FRAME_WIDTH = 800         # Downscale for ultra-high FPS (Pi 4 handles 800x600 easily)
FRAME_HEIGHT = 600

# Region of Interest (ROI) - Adjust these values to crop out surrounding background structures
# Values represent fraction of the image height/width: (ymin, ymax, xmin, xmax)
ROI_BOUNDS = (0.1, 0.9, 0.1, 0.9) 

# Detection Settings
ORIENTATION = 'horizontal'  # 'horizontal' or 'vertical'
MIN_ROD_SPACING = 20        # Minimum expected distance between adjacent parallel rods in pixels
MIN_ROD_LENGTH = 150        # Minimum expected length of a physical rod segment in pixels

# GPIO Setup
ALERT_PIN = 18              # GPIO pin to trigger buzzer/relay/PLC (Physical Pin 12)
EXPECTED_ROD_COUNT = 26     # Trigger alert if the count deviates from this
# =======================================================

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

def setup_gpio():
    if not GPIO_AVAILABLE:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(ALERT_PIN, GPIO.OUT)
    GPIO.output(ALERT_PIN, GPIO.LOW)
    print(f"GPIO Setup Complete. Alert Pin: GPIO {ALERT_PIN}")

def trigger_alert(state):
    if GPIO_AVAILABLE:
        GPIO.output(ALERT_PIN, GPIO.HIGH if state else GPIO.LOW)

def log_count(count):
    log_file = "rod_counts_log.csv"
    file_exists = os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "RodCount", "Status"])
        status = "OK" if count == EXPECTED_ROD_COUNT else "ERROR"
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), count, status])

def process_frame(frame):
    h, w, _ = frame.shape
    
    # 1. Extract Region of Interest (ROI) to ignore outer background edges
    ymin, ymax = int(h * ROI_BOUNDS[0]), int(h * ROI_BOUNDS[1])
    xmin, xmax = int(w * ROI_BOUNDS[2]), int(w * ROI_BOUNDS[3])
    roi = frame[ymin:ymax, xmin:xmax]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 2. White Top-Hat Morphological Filter (Clears shadows/uneven light)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # 3. Canny Edge Extraction
    edges = cv2.Canny(tophat, 50, 150)
    
    # 4. Hough Lines Extraction
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=50, maxLineGap=10)
    
    filtered_lines = []
    if raw_lines is not None:
        for line in raw_lines:
            _, angle, _ = get_line_props(line[0])
            if ORIENTATION == 'horizontal':
                if angle < 15 or angle > 165:
                    filtered_lines.append(line)
            else:
                if abs(angle - 90) < 15:
                    filtered_lines.append(line)
                    
    # 5. Collinear Merging
    collinear_merged = merge_collinear_lines(filtered_lines, max_angle_diff=8, max_rho_diff=15, max_gap=150)
    
    # 6. Parallel Suppression (Thickness merging)
    final_rods = merge_parallel_lines(collinear_merged, orientation=ORIENTATION, max_dist=MIN_ROD_SPACING)
    
    # Filter short noise lines
    final_rods = [l for l in final_rods if math.hypot(l[2]-l[0], l[3]-l[1]) >= MIN_ROD_LENGTH]
    
    # Draw centerline detections strictly between endpoints on output ROI
    output_roi = roi.copy()
    for i, line in enumerate(final_rods):
        x1, y1, x2, y2 = line
        cv2.line(output_roi, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(output_roi, (x1, y1), 5, (0, 255, 0), -1)
        cv2.circle(output_roi, (x2, y2), 5, (0, 255, 0), -1)
        
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        cv2.putText(output_roi, str(i + 1), (mid_x, mid_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            
    # Reassemble ROI into frame for visual feedback
    output_frame = frame.copy()
    output_frame[ymin:ymax, xmin:xmax] = output_roi
    
    # Draw ROI boundaries
    cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
    
    return output_frame, len(final_rods)

def main():
    setup_gpio()
    
    print("Starting Video Stream...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Allow camera sensor to warm up
    time.sleep(1.5)
    
    last_log_time = 0
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            # Process Frame
            start_t = time.perf_counter()
            processed_frame, count = process_frame(frame)
            proc_time_ms = (time.perf_counter() - start_t) * 1000
            
            # FPS Calculation
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
                
            # Trigger Physical Alert if Rod Count is Incorrect
            if count != EXPECTED_ROD_COUNT:
                trigger_alert(True)
                status_color = (0, 0, 255) # Red
                status_text = f"ALERT: Count Mismatch! Expected {EXPECTED_ROD_COUNT}"
            else:
                trigger_alert(False)
                status_color = (0, 255, 0) # Green
                status_text = "Status: OK"
                
            # Log count to CSV every 5 seconds
            if time.time() - last_log_time > 5.0:
                log_count(count)
                last_log_time = time.time()
                
            # Overlay visual information
            cv2.putText(processed_frame, f"Detected Rods: {count}", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(processed_frame, status_text, (30, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(processed_frame, f"FPS: {fps} | Latency: {proc_time_ms:.1f}ms", (30, FRAME_HEIGHT - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("Raspberry Pi 4 Rod Counter - Real Time", processed_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping loop...")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        print("Cleaned up resources. Process complete.")

if __name__ == "__main__":
    main()
