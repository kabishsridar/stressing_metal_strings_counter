"""
Real-time YOLOv8-Segmentation Rod Counter.
Loads a custom-trained yolov8n-seg model, segmenting and counting metal rods with pixel-perfect accuracy.
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

# Try importing Ultralytics for YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Ultralytics library not found. Install via: pip install ultralytics")

# ==================== CONFIGURATION ====================
MODEL_PATH = "weights/best.pt"  # Path to your trained weights (or 'weights/best.onnx' on RPi)
CAMERA_INDEX = 0                # 0 for default Pi camera / USB webcam
FRAME_WIDTH = 640               # YOLO operates natively at 640x640 resolution
FRAME_HEIGHT = 480

# Region of Interest (ROI) - optional, keeps the frame focused and boosts FPS
USE_ROI = False
ROI_BOUNDS = (0.1, 0.9, 0.1, 0.9)  # (ymin, ymax, xmin, xmax) fractions

# GPIO Setup
ALERT_PIN = 18                  # GPIO pin to trigger buzzer/relay/PLC (Physical Pin 12)
EXPECTED_ROD_COUNT = 26         # Target rod count
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
    log_file = "yolo_rod_counts_log.csv"
    file_exists = os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "RodCount", "Status"])
        status = "OK" if count == EXPECTED_ROD_COUNT else "ERROR"
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), count, status])

def process_frame_yolo(frame, model):
    """Segment and count rods using YOLOv8 instance segmentation"""
    h, w, _ = frame.shape
    
    # 1. Apply ROI if enabled
    if USE_ROI:
        ymin, ymax = int(h * ROI_BOUNDS[0]), int(h * ROI_BOUNDS[1])
        xmin, xmax = int(w * ROI_BOUNDS[2]), int(w * ROI_BOUNDS[3])
        process_img = frame[ymin:ymax, xmin:xmax]
    else:
        ymin, ymax, xmin, xmax = 0, h, 0, w
        process_img = frame
        
    # 2. Run Inference
    # verbose=False keeps the console clean and speeds up loop slightly
    results = model.predict(process_img, conf=0.4, verbose=False)
    result = results[0]
    
    # 3. Extract segmented masks
    masks = result.masks
    rod_count = len(masks) if masks is not None else 0
    
    # Draw perfect semi-transparent overlays on detected rods
    output_img = process_img.copy()
    
    if masks is not None:
        # Create a blank overlay canvas for mask blending
        overlay = output_img.copy()
        
        for i, mask in enumerate(masks.xy):
            # Convert polygon coordinates to numpy int32 format
            pts = np.array(mask, dtype=np.int32)
            
            # Fill the rod polygon with transparent cyan color (0, 255, 255)
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            
            # Draw a solid green outline around the rod boundaries
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Draw labels at the center of the rod segment
            if len(pts) > 0:
                mid_pt = pts[len(pts) // 2]
                cv2.putText(overlay, f"#{i + 1}", (mid_pt[0], mid_pt[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
        # Blend the overlay with the original image (alpha = 0.6)
        cv2.addWeighted(overlay, 0.4, output_img, 0.6, 0, output_img)
        
    # Reassemble ROI if enabled
    output_frame = frame.copy()
    if USE_ROI:
        output_frame[ymin:ymax, xmin:xmax] = output_img
        cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
    else:
        output_frame = output_img
        
    return output_frame, rod_count

def main():
    if not YOLO_AVAILABLE:
        print("Please install Ultralytics package: pip install ultralytics")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Custom trained weights not found at: {MODEL_PATH}")
        print("Please copy your trained 'best.pt' or 'best.onnx' model to that directory.")
        return
        
    setup_gpio()
    
    print(f"Loading custom YOLOv8 model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("Model Loaded successfully.")
    
    print("Starting Deep Learning Video Stream...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Warmup
    time.sleep(1.5)
    
    last_log_time = 0
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                break
                
            # Process Frame using deep learning
            start_t = time.perf_counter()
            processed_frame, count = process_frame_yolo(frame, model)
            proc_time_ms = (time.perf_counter() - start_t) * 1000
            
            # FPS Tracker
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
                
            # Hardware alarm control
            if count != EXPECTED_ROD_COUNT:
                trigger_alert(True)
                status_color = (0, 0, 255)
                status_text = f"ALERT: Count Mismatch! Expected {EXPECTED_ROD_COUNT}"
            else:
                trigger_alert(False)
                status_color = (0, 255, 0)
                status_text = "Status: OK"
                
            # Log results to CSV
            if time.time() - last_log_time > 5.0:
                log_count(count)
                last_log_time = time.time()
                
            # Add visual HUD
            cv2.putText(processed_frame, f"YOLOv8 Rods: {count}", (30, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(processed_frame, status_text, (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(processed_frame, f"FPS: {fps} | Latency: {proc_time_ms:.1f}ms", 
                        (30, FRAME_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Render Window
            cv2.imshow("Stressing Metal Strings - YOLOv8 Instance Segmentation", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nHalting counter...")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        print("Resources cleaned. System offline.")

if __name__ == "__main__":
    main()
