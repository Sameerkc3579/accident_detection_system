import os
import cv2
import shutil
import uuid
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="Sentinel AI Backend")

# Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "FINAL_BEST_ACCIDENT_MODEL.pt")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure static directory exists
os.makedirs(STATIC_DIR, exist_ok=True)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load YOLO Model
try:
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.post("/detect")
async def detect_accident(file: UploadFile = File(...)):
    if model is None:
        return {"status": "Error", "message": "Model not loaded. Check backend console."}

    # Generate unique filenames
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    input_path = os.path.join(STATIC_DIR, f"input_{filename}")
    output_filename = f"processed_{filename}"
    output_path = os.path.join(STATIC_DIR, output_filename)

    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process Video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {"status": "Error", "message": "Could not open video file."}

    # Video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    # Resize for performance if video is too large (limiting to 640 width)
    # This prevents "buffering" / timeout on large videos
    target_width = original_width
    target_height = original_height
    if original_width > 640:
        scale_ratio = 640 / original_width
        target_width = 640
        target_height = int(original_height * scale_ratio)

    # Output writer - switching to mp4v as it is more reliable on standard Windows OpenCV installs
    # Note: If this video does not play in Chrome, you may need to download it or use VLC.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    if not out.isOpened():
        print("Error: Could not initialize video writer with mp4v.")
        return {"status": "Error", "message": "Server video codec error. Check logs."}

    frame_count = 0
    total_confidence = 0
    detections_count = 0
    accident_detected_frames = 0
    
    total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {filename} ({target_width}x{target_height}) - {total_frames_est} frames")

    # Speed Optimization: Skip frames
    # Processing every frame on CPU is too slow (1-5 FPS).
    # We will detect every 5th frame (approx every 0.15s) and persist the boxes.
    SKIP_FRAMES = 5
    last_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame if needed
        if target_width != original_width:
            frame = cv2.resize(frame, (target_width, target_height))

        # Run YOLO inference ONLY every N frames
        if frame_count % SKIP_FRAMES == 0:
            results = model.predict(frame, conf=0.6, verbose=False)
            last_results = results
        
        # Draw results (using current or last known detection)
        if last_results:
            res_plotted = last_results[0].plot(img=frame)
            
            # Analyze detections (only count unique detection events to avoid overcounting)
            if frame_count % SKIP_FRAMES == 0:
                boxes = last_results[0].boxes
                if len(boxes) > 0:
                    detections_count += len(boxes)
                    conf_sum = sum(box.conf.item() for box in boxes)
                    total_confidence += conf_sum
                    accident_detected_frames += 1
        else:
            res_plotted = frame

        out.write(res_plotted)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{total_frames_est} frames...")

        # Safety break for very long videos to prevent infinite hanging in demo
        if frame_count > 1800: # Limit to ~1 minute @ 30fps
            print("Video too long, truncating...")
            break

    cap.release()
    out.release()
    
    # Calculate metrics
    avg_conf = 0.0
    if detections_count > 0:
        avg_conf = total_confidence / detections_count
    
    status = "Normal Traffic"
    if accident_detected_frames > 0:
        status = "Accident Detected"

    video_url = f"http://localhost:8000/static/{output_filename}"

    return {
        "status": status,
        "video_url": video_url,
        "confidence": float(f"{avg_conf:.2f}")
    }
