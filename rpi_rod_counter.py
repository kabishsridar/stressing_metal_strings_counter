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
ORIENTATION = 'horizontal'  # 'horizontal' (rows sum) or 'vertical' (cols sum)
MIN_ROD_SPACING = 15        # Minimum expected distance between adjacent rods in pixels
PEAK_THRESHOLD_RATIO = 0.18 # Fraction of max peak intensity to register as a rod
SMOOTHING_WINDOW = 11       # Moving average window size for peak smoothing

# GPIO Setup
ALERT_PIN = 18              # GPIO pin to trigger buzzer/relay/PLC (Physical Pin 12)
EXPECTED_ROD_COUNT = 26     # Trigger alert if the count deviates from this
# =======================================================

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
    
    # 2. White Top-Hat Morphological Filter (Clears large shadows/illumination slopes)
    # RPi 4 is fast enough to do 15x15 rect morphology in < 1ms on 800x600
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # 3. 1D Projection
    axis = 1 if ORIENTATION == 'horizontal' else 0
    profile = np.mean(tophat, axis=axis)
    
    # 4. Smoothing to reject pixel noise
    smoothed = np.convolve(profile, np.ones(SMOOTHING_WINDOW)/SMOOTHING_WINDOW, mode='same')
    
    # 5. Peak Finding
    peaks = []
    max_val = np.max(smoothed)
    threshold = max_val * PEAK_THRESHOLD_RATIO if max_val > 10 else 255 # Avoid picking noise in total dark
    
    for i in range(MIN_ROD_SPACING, len(smoothed) - MIN_ROD_SPACING):
        if smoothed[i] > threshold:
            neighborhood = smoothed[i - MIN_ROD_SPACING : i + MIN_ROD_SPACING + 1]
            if smoothed[i] == np.max(neighborhood):
                if len(peaks) == 0 or (i - peaks[-1]) >= MIN_ROD_SPACING:
                    peaks.append(i)
                    
    # Draw detections on output
    output_roi = roi.copy()
    for idx in peaks:
        if ORIENTATION == 'horizontal':
            cv2.line(output_roi, (0, idx), (roi.shape[1], idx), (0, 0, 255), 2)
        else:
            cv2.line(output_roi, (idx, 0), (idx, roi.shape[0]), (0, 0, 255), 2)
            
    # Reassemble ROI into frame for visual feedback
    output_frame = frame.copy()
    output_frame[ymin:ymax, xmin:xmax] = output_roi
    
    # Draw ROI boundaries
    cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
    
    return output_frame, len(peaks)

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
